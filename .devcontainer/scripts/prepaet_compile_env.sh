# 更换镜像源,并添加llvm的源
cat >/etc/apt/sources.list <<EOF
deb https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse

# deb https://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse

deb https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
EOF
apt update -y

# 避免安装包时时区的设置阻塞dev容器的启动
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata

# 配置llvm镜像源
apt install -y wget curl

wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
cat <<'EOF' > /etc/apt/sources.list.d/llvm-apt.list
deb https://mirrors.cernet.edu.cn/llvm-apt/focal/ llvm-toolchain-focal-16 main
EOF

# 安装编译tch-rs所需的llvm相关工具链,cmake默认容器中已经安装
apt update -y
# 安装clang与lld,以及clangd插件所需要的clangd lsp server
apt install -y clang-12 clangd-12 lld-12 clang-16 lld-16 
# 设置clangd-12为默认可执行的clangd
update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100

# 安装rust
export RUSTUP_DIST_SERVER="https://rsproxy.cn"
export RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- --default-toolchain stable -y