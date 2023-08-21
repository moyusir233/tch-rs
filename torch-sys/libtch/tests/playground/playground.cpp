#include <ATen/ops/tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <torch_api.h>

int main() {
  torch::Tensor x = torch::tensor({1, 2, 3}, torch::TensorOptions()
                                                 .device("cuda:0")
                                                 .dtype(c10::ScalarType::Float)
                                                 .requires_grad(true));
  x.register_hook([](const torch::Tensor &tensor) {
    printf("tensor hook is invoked:\n");
    tensor.print();
  });

  auto y = (x * x).sum();
  y.print();
  y.backward();

  x.grad().print();
}