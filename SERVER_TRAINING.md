# 服务器训练指南 - RTX 4090 24GB

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

### RTX 4090 24GB 推荐配置

| 模型 | Batch Size | 梯度累积 | 等效 Batch | 序列长度 | 预计速度 |
|------|-----------|---------|-----------|---------|----------|
| Qwen2.5-Coder-1.5B | 24 | 4 | 96 | 512 | ~0.8s/it |
| Qwen2.5-Coder-3B | 12 | 8 | 96 | 512 | ~1.5s/it |
| Qwen2.5-Coder-7B | 4 | 16 | 64 | 256 | ~3s/it |

### 自定义参数

编辑 `train_server.sh` 顶部的配置：

```bash
# ==================== 配置参数 ====================
CONDA_ENV_NAME="gptfuzzer"
PYTHON_VERSION="3.10"
ATTACK_TYPE="sqli"               # sqli, xss, rce
MODEL_PRESET="qwen2.5-coder-3b"  # 推荐 3B 模型
PRETRAIN_EPOCHS=3
REWARD_SAMPLES=4000
RL_EPISODES=20
BATCH_SIZE=12                    # 3B 模型适配 24GB
GRADIENT_ACCUMULATION=8
MAX_LENGTH=512                   # 更长序列
```

## 训练流程

脚本会自动执行以下步骤：

1. **环境检查** - 检查 GPU、CUDA、Conda
2. **创建虚拟环境** - 安装 Python 3.10
3. **安装依赖** - PyTorch CUDA 12.4、Transformers、Flash Attention 2
4. **Stage 1: 预训练** - 学习攻击语法
5. **Stage 2: 奖励模型** - 学习 WAF 判断
6. **Stage 3: RL (PPO)** - 优化绕过能力
7. **测试** - 生成测试 payload

## 预计训练时间 (RTX 4090 24GB)

| 阶段 | 数据量 | 预计时间 |
|------|--------|----------|
| 预训练 (3B) | 512K samples, 3 epochs | 6-8 小时 |
| 奖励模型 | 4K samples, 4 epochs | 20-40 分钟 |
| RL (PPO) | 20 episodes | 1.5-3 小时 |
| **总计** | | **8-12 小时** |

**注**: 使用 1.5B 模型可缩短约 40% 时间

## RTX 4090 优势

- **Ada Lovelace 架构 (SM 89)**: PyTorch 完美支持，无兼容性问题
- **Flash Attention 2**: 原生支持，加速约 2x
- **24GB VRAM**: 可运行 7B 模型 (带梯度检查点)
- **成熟稳定**: CUDA 12.4 完全兼容

## 常见问题

### Q1: HuggingFace 下载慢

脚本已自动配置镜像。如需手动设置：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Q2: Flash Attention 安装失败

```bash
# 手动安装
pip install flash-attn --no-build-isolation

# 如果失败，脚本会自动回退到标准注意力机制
# 不影响训练，只是稍慢一些
```

### Q3: CUDA 版本检查

```bash
nvidia-smi         # 查看驱动版本
nvcc --version     # 查看 CUDA 版本
python -c "import torch; print(torch.cuda.is_available())"
```

RTX 4090 推荐 CUDA 12.1+

### Q4: WAF 服务未运行

奖励模型训练需要 WAF 服务来生成标注数据：

```bash
# 启动 WAF (Docker)
docker-compose up -d modsecurity

# 或使用已有标注数据
# 将数据放入 data/labeled/ 目录
```

### Q5: 显存不足 (OOM)

如果遇到 OOM，尝试：
```bash
# 减小 batch size
BATCH_SIZE=8

# 启用梯度检查点 (在 config.py 中设置)
"gradient_checkpointing": True
```

## 输出文件

训练完成后，模型保存在：

```
models/
├── pretrain_sqli_qwen2_5_coder_3b/   # 预训练模型
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

| 特性 | Windows (RTX 4070 8GB) | 服务器 (RTX 4090 24GB) |
|------|------------------------|------------------------|
| 推荐模型 | Qwen2.5-Coder-0.5B | Qwen2.5-Coder-3B |
| Batch Size | 16 | 12-24 |
| 序列长度 | 128 | 512 |
| Flash Attention | 不支持 | 支持 (2x 加速) |
| DataLoader Workers | 2 | 8 |
| 预训练时间 (3 epochs) | 30-50 小时 | 6-8 小时 |

## 模型选择建议

| 目标 | 推荐模型 | 说明 |
|------|---------|------|
| 快速实验 | qwen2.5-coder-1.5b | 训练快，效果不错 |
| 生产使用 | qwen2.5-coder-3b | 平衡速度和质量 |
| 最佳效果 | qwen2.5-coder-7b | 需要梯度检查点 |
