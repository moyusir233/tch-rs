bash /root/scripts/prepaet_compile_env.sh

# 配置git
apt install -y git
git config --global user.name moyu
git config --global user.email 279972161@qq.com

# Clean up
rm -rf /var/lib/apt/lists/*