use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tch::Tensor;

#[inline]
fn try_cuda_graph_bench(c: &mut Criterion) -> anyhow::Result<()> {
    use tch::nn::OptimizerConfig;
    use tch::IndexOp;

    let dataset = tch::vision::dataset::Dataset {
        train_images: Tensor::rand([128 * 20, 3, 256, 256], (tch::Kind::Float, tch::Device::Cpu)),
        train_labels: Tensor::rand([128 * 20], (tch::Kind::Float, tch::Device::Cpu)),
        test_images: Tensor::new(),
        test_labels: Tensor::new(),
        labels: 0,
    };
    let device = tch::Device::Cuda(
        std::env::var("CUDA_TEST_DEVICE").map(|device| device.parse().unwrap()).unwrap_or(0),
    );
    let var_store = tch::nn::VarStore::new(device);
    let model = tch::vision::resnet::resnet50(&var_store.root(), 10);
    let mut optim = tch::nn::Sgd::default().build(&var_store, 0.01)?;
    let batch_size = 16;

    let mut static_data = dataset.train_images.i(0..batch_size).to_device(device);
    let mut static_label = dataset.train_labels.i(0..batch_size).to_device(device);
    let mut static_loss = tch::Tensor::empty([1, 1], (tch::Kind::Double, device));

    // warm up
    use tch::cuda_guards::{CUDAStreamGuard, GuardScope};
    use tch::cuda_stream::{CUDAStream, CUDAStreamExt};
    let capture_stream = CUDAStream::get_stream_from_pool(false, device.index() as i8)?;
    {
        capture_stream.wait_stream(
            CUDAStream::get_current_cuda_stream(device.index() as i8)?.as_ref().unwrap(),
        )?;
        CUDAStreamGuard::scope(capture_stream.as_ref().unwrap(), || {
            for _ in 0..5 {
                optim.zero_grad();
                static_data
                    .apply_t(&model, true)
                    .cross_entropy_for_logits(&static_label.to_kind(tch::Kind::Int64))
                    .backward();
                optim.step();
            }
            optim.zero_grad();
            Ok(())
        })?;
        CUDAStream::get_current_cuda_stream(device.index() as i8)?.wait_stream(&capture_stream)?;
    }

    let mut group = c.benchmark_group("cuda_graph_bench");
    group.sample_size(10);

    use tch::cuda_graph::{CUDAGraph, CUDAGraphExt};
    {
        let mut graph = CUDAGraph::new();

        // capture
        graph.as_mut().unwrap().record(Some(&capture_stream), None, || {
            let loss = static_data
                .apply_t(&model, true)
                .cross_entropy_for_logits(&static_label.to_kind(tch::Kind::Int64));
            static_loss.copy_(&loss);
            optim.backward_step(&loss);
            Ok(())
        })?;

        // benchmark replay
        group.bench_function("train_resnet_cifar10_by_cuda_graph", |b| {
            b.iter(|| {
                for (data, label) in dataset.train_iter(batch_size).to_device(device) {
                    static_data.copy_(&data);
                    static_label.copy_(&label);
                    graph.as_mut().unwrap().replay().unwrap();
                }
                println!("loss:{}", black_box(f64::try_from(static_loss.shallow_clone()).unwrap()));
            });
        });
    }

    // 不使用图进行训练
    group.bench_function("train_resnet_cifar10", |b| {
        b.iter(|| {
            for (data, label) in dataset.train_iter(batch_size).to_device(device) {
                static_loss = data
                    .apply_t(&model, true)
                    .cross_entropy_for_logits(&label.to_kind(tch::Kind::Int64));
                optim.backward_step(&static_loss);
            }
            println!("loss:{}", black_box(f64::try_from(static_loss.shallow_clone()).unwrap()));
        });
    });

    group.finish();
    Ok(())
}

fn cuda_graph(c: &mut Criterion) {
    try_cuda_graph_bench(c).unwrap()
}

criterion_group!(cuda_graph_bench, cuda_graph);
criterion_main!(cuda_graph_bench);
