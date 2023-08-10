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

DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata
apt install -y wget curl git
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
cat <<'EOF' > /etc/apt/sources.list.d/llvm-apt.list
deb https://mirrors.cernet.edu.cn/llvm-apt/focal/ llvm-toolchain-focal-16 main
EOF
apt update -y
# 安装clang与lld
apt install -y clang-12 lld-12 clang-16 lld-16 

cat >>/root/.bashrc <<EOF
if [ -e "/root/.cargo/env" ]; then
    echo "cargo has been install"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
fi
EOF

# ssh配置
cat > /etc/ssh/sshd_config <<EOF
Include /etc/ssh/sshd_config.d/*.conf

Port 2222
PermitRootLogin yes
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding yes
PrintMotd no
AcceptEnv LANG LC_*
Subsystem	sftp	/usr/lib/openssh/sftp-server
# custom
PubkeyAuthentication yes    # 启用公告密钥配对认证方式 
PasswordAuthentication no   # 禁止密码验证登录,如果启用的话,RSA认证登录就没有意义了
PermitRootLogin yes
EOF
service ssh restart