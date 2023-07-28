FROM rust as builder

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# 将rust工具链拷贝到镜像中
COPY --from=builder /usr/local/cargo /root/.cargo
RUN echo "source /root/.cargo/env" >> /root/.bashrc
