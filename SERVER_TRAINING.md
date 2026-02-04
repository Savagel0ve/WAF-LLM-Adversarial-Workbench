# 服务器训练指南 - RTX 5090 32GB

## 快速开始

### 1. 上传代码到服务器

```bash
# 在本地执行
scp -r WAF-LLM-Adversarial-Workbench user@server:/path/to/
```

### 2. 一键训练

```bash
# SSH 到服务器
ssh user@server

# 进入项目目录
cd /path/to/WAF-LLM-Adversarial-Workbench

# 添加执行权限
chmod +x train_server.sh

# 运行一键训练脚本
./train_server.sh
```

## 训练参数说明

### RTX 5090 32GB 推荐配置

| 模型 | Batch Size | 梯度累积 | 等效 Batch | 预计速度 |
|------|-----------|---------|-----------|----------|
| Qwen2.5-Coder-1.5B | 32 | 4 | 128 | ~0.5s/it |
| Qwen2.5-Coder-3B | 16 | 8 | 128 | ~1s/it |
| Qwen2.5-Coder-7B (4-bit) | 4 | 32 | 128 | ~2s/it |

### 自定义参数

编辑 `train_server.sh` 顶部的配置：

```bash
# ==================== 配置参数 ====================
CONDA_ENV_NAME="gptfuzzer"
PYTHON_VERSION="3.10"
ATTACK_TYPE="sqli"           # sqli, xss, rce
MODEL_PRESET="qwen2.5-coder-1.5b"  # 或 qwen2.5-coder-3b
PRETRAIN_EPOCHS=3
REWARD_SAMPLES=4000
RL_EPISODES=20
BATCH_SIZE=32
GRADIENT_ACCUMULATION=4
```

## 训练流程

脚本会自动执行以下步骤：

1. **环境检查** - 检查 GPU、CUDA、Conda
2. **创建虚拟环境** - 安装 Python 3.10
3. **安装依赖** - PyTorch、Transformers、Flash Attention 等
4. **Stage 1: 预训练** - 学习攻击语法
5. **Stage 2: 奖励模型** - 学习 WAF 判断
6. **Stage 3: RL (PPO)** - 优化绕过能力
7. **测试** - 生成测试 payload

## 预计训练时间 (RTX 5090 32GB)

| 阶段 | 数据量 | 预计时间 |
|------|--------|----------|
| 预训练 | 512K samples, 3 epochs | 4-6 小时 |
| 奖励模型 | 4K samples, 4 epochs | 15-30 分钟 |
| RL (PPO) | 20 episodes | 1-2 小时 |
| **总计** | | **6-9 小时** |

## 常见问题

### Q1: HuggingFace 下载慢

使用镜像：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Q2: CUDA 版本不匹配

检查 CUDA 版本：
```bash
nvidia-smi
nvcc --version
```

RTX 5090 需要 CUDA 12.4+

### Q3: Flash Attention 安装失败

```bash
# 手动安装
pip install flash-attn --no-build-isolation

# 如果失败，可以禁用 Flash Attention
# 在训练命令中移除 --flash-attention 参数
```

### Q4: WAF 服务未运行

奖励模型训练需要 WAF 服务来生成标注数据：

```bash
# 启动 WAF (Docker)
docker-compose up -d modsecurity

# 或使用已有标注数据
# 将数据放入 data/labeled/ 目录
```

## 输出文件

训练完成后，模型保存在：

```
models/
├── pretrain_sqli_qwen2_5_coder_1_5b/  # 预训练模型
├── reward_sqli_qwen/
│   └── final_reward_model/            # 奖励模型
└── rl_sqli_qwen/
    └── final_model/                   # 最终 RL 模型
```

## 后续使用

```bash
# 生成 payload
python train/generate_payloads.py --model_path models/rl_sqli_qwen/final_model

# 测试模型
python train/test_rl_model.py --model_path models/rl_sqli_qwen/final_model --num_samples 100

# 评估绕过率
python train/evaluate_rl.py --model_path models/rl_sqli_qwen/final_model --waf_url http://localhost:8081
```

## 与 Windows 版本对比

| 特性 | Windows (RTX 4070 8GB) | 服务器 (RTX 5090 32GB) |
|------|------------------------|------------------------|
| 推荐模型 | Qwen2.5-Coder-0.5B | Qwen2.5-Coder-1.5B/3B |
| Batch Size | 16 | 32-64 |
| Flash Attention | 不支持 | 支持 |
| DataLoader Workers | 2 | 8 |
| 预训练时间 | 30-50 小时 | 4-6 小时 |
