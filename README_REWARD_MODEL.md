# 奖励模型训练指南

这是基于 GPTFuzzer 论文的奖励模型训练实现。奖励模型用于预测 payload 绕过 WAF 的概率，是强化学习阶段的关键组件。

## 📋 目录

- [概述](#概述)
- [训练流程](#训练流程)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
- [论文细节](#论文细节)
- [常见问题](#常见问题)

## 概述

### 什么是奖励模型？

奖励模型是一个 **GPT-2 序列分类模型**，用于预测攻击 payload 绕过 WAF 的概率：

- **输入**: 攻击 payload 文本
- **输出**: 绕过概率 `r(τ) ∈ [0, 1]`
- **架构**: 预训练 GPT-2 + 分类头 (线性层 + Sigmoid)
- **损失函数**: BCEWithLogitsLoss

### 为什么需要奖励模型？

在强化学习（PPO）阶段，奖励模型提供比简单二值奖励（通过/拦截）更丰富的反馈信号：

1. **梯度信号**: 提供连续的概率值而非离散的 0/1
2. **高效性**: 避免在训练过程中频繁请求真实 WAF
3. **稳定性**: 减少网络延迟和 WAF 状态变化的影响

## 训练流程

奖励模型训练分为两个步骤：

```
预训练模型
    ↓
【步骤1】生成标记数据 (generate_labeled_data.py)
    ├─ 从预处理数据中采样
    ├─ 发送到 WAF 测试
    ├─ 根据 WAF 响应打标签
    └─ 保存为 CSV
    ↓
【步骤2】训练分类模型 (train_reward_model.py)
    ├─ 加载预训练模型
    ├─ 替换分类头
    ├─ 训练 4 epochs
    └─ 保存奖励模型
    ↓
奖励模型 (用于 PPO)
```

## 快速开始

### 前置条件

1. ✅ 完成预训练（参考 `GPTFuzzer 预训练复现指南.md`）
2. ✅ WAF 服务正在运行（默认: `http://localhost:8081`）
3. ✅ 有预处理的 payload 数据

### SQLi 奖励模型训练

```powershell
# 一键训练 SQLi 奖励模型
.\train_reward_sqli.ps1
```

这个脚本会：
1. 从 SQLi 数据中采样 4000 条
2. 通过 WAF 测试并打标签
3. 训练 GPT-2 分类模型（4 epochs）
4. 在测试集上评估

### XSS 奖励模型训练

```powershell
# 一键训练 XSS 奖励模型
.\train_reward_xss.ps1
```

### 自定义训练

如果需要更细粒度的控制：

```powershell
# 步骤1: 生成标记数据
python train\generate_labeled_data.py `
    --attack_type sqli `
    --input_file .\data\processed\sqli\train.txt `
    --output_dir .\data\labeled `
    --num_samples 4000 `
    --waf_url http://localhost:8081 `
    --balance_ratio 0.5

# 步骤2: 训练模型
python train\train_reward_model.py `
    --pretrained_model_path .\models\pretrain_sqli_gpt2_small `
    --data_path .\data\labeled `
    --output_dir .\models\reward_sqli `
    --batch_size 32 `
    --learning_rate 2e-5 `
    --epochs 4 `
    --fp16

# 步骤3: 测试模型
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload "' OR 1=1 --"
```

## 详细说明

### 步骤1: 生成标记数据

#### 数据采样数量（论文设定）

- **SQLi**: 4,000 条
- **XSS**: 2,000 条
- **RCE**: 2,000 条

#### 标签定义

| 标签 | 含义 | WAF 响应 |
|------|------|----------|
| `1` | 绕过 (Bypassing) | 200 OK (未拦截) |
| `0` | 拦截 (Blocked) | 403 Forbidden (拦截) |

#### 数据平衡

脚本会自动平衡正负样本比例（默认 1:1），避免类别不平衡。

#### 输出格式

CSV 文件示例：

```csv
text,label
"' OR 1=1 --",1
"UNION SELECT * FROM users",0
"1' AND SLEEP(5)--",1
```

### 步骤2: 训练分类模型

#### 超参数（论文设定）

| 参数 | 值 | 说明 |
|------|-----|------|
| `epochs` | 4 | 训练轮数 |
| `batch_size` | 32 | 批次大小 |
| `learning_rate` | 2e-5 | 学习率 |
| `warmup_ratio` | 0.1 | 预热比例 |
| `weight_decay` | 0.01 | 权重衰减 |
| `max_length` | 128 | 最大序列长度 |

#### 模型架构

```
GPT-2 Transformer (预训练)
    ↓
[CLS] Token 的隐藏状态
    ↓
Linear Layer (hidden_size → 1)
    ↓
BCEWithLogitsLoss
    ↓
Sigmoid → 概率 [0, 1]
```

#### 评估指标

- **Accuracy**: 预测准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1 分数
- **AUC-ROC**: ROC 曲线下面积

论文目标（ModSecurity）：
- SQLi: AUC > 99%
- XSS: AUC > 98%

### 步骤3: 测试模型

#### 单个 Payload 测试

```powershell
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload "' OR 1=1 --"
```

输出：
```
绕过概率: 0.9234
🟢 高概率绕过
```

#### 批量测试

```powershell
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload_file .\data\processed\sqli\test.txt
```

#### 交互式测试

```powershell
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model
```

## 论文细节

### 与 GPTFuzzer 的对应

本实现严格遵循论文设定：

1. **数据量**:
   - SQLi: 4,000 条 ✅
   - XSS/RCE: 2,000 条 ✅

2. **超参数**:
   - Epochs: 4 ✅
   - Batch Size: 32 ✅
   - Learning Rate: 2e-5 ✅
   - Warmup: 10% ✅

3. **损失函数**:
   - BCEWithLogitsLoss ✅

4. **数据划分**:
   - Train: 70% ✅
   - Val: 15% ✅
   - Test: 15% ✅

### 改进点

相比论文，本实现增加了：

1. **数据验证**: 使用 `verifier.py` 过滤无效 payload
2. **数据平衡**: 自动平衡正负样本
3. **早停机制**: 可选的早停（避免过拟合）
4. **混合精度**: FP16 加速训练
5. **实时监控**: TensorBoard 支持

## 常见问题

### Q1: 如果没有真实 WAF 怎么办？

使用模拟 WAF：

```python
from waf_env import MockWAFEnvironment

waf = MockWAFEnvironment(block_rate=0.7)
```

### Q2: 数据量不够怎么办？

可以减少采样数量：

```powershell
python train\generate_labeled_data.py `
    --num_samples 1000  # 降低到 1000
```

但可能影响模型性能。

### Q3: 训练时显存不足？

降低 batch size：

```powershell
python train\train_reward_model.py `
    --batch_size 16  # 降低到 16
```

或使用梯度累积（需修改代码）。

### Q4: 如何提高模型性能？

1. **增加数据量**: 采样更多 payload
2. **数据增强**: 使用 payload 变异
3. **调整超参数**: 学习率、epoch 数
4. **更大模型**: 使用 GPT-2 Medium/Large

### Q5: 如何使用训练好的模型？

在 PPO 训练中：

```python
from test_reward_model import RewardModelInference

# 加载模型
reward_model = RewardModelInference(
    model_path="./models/reward_sqli/final_reward_model"
)

# 预测
prob = reward_model.predict_single("' OR 1=1 --")
print(f"绕过概率: {prob}")
```

## 文件结构

```
train/
├── generate_labeled_data.py  # 数据标记脚本
├── train_reward_model.py     # 模型训练脚本
├── test_reward_model.py      # 模型测试脚本
├── waf_env.py                # WAF 环境接口
├── verifier.py               # Payload 验证器
└── reward_model.py           # 奖励函数（用于 PPO）

train_reward_sqli.ps1         # SQLi 一键训练脚本
train_reward_xss.ps1          # XSS 一键训练脚本

data/
├── labeled/                  # 标记数据
│   ├── sqli_train.csv
│   ├── sqli_val.csv
│   └── sqli_test.csv
└── processed/                # 预处理数据

models/
├── reward_sqli/              # SQLi 奖励模型
│   ├── final_reward_model/
│   └── logs/
└── reward_xss/               # XSS 奖励模型
```

## 下一步

训练完成后，奖励模型用于：

1. **PPO 强化学习**: 作为奖励函数
2. **Payload 评估**: 快速预测绕过概率
3. **数据筛选**: 过滤低质量 payload

参考 `WAF 绕过奖励模型训练指南.md` 了解如何进入 PPO 阶段。

## 参考资料

- [GPTFuzzer 论文](https://arxiv.org/abs/2309.10253)
- [GPTFuzzer 论文细节总结.md](./GPTFuzzer 论文细节总结.md)
- [WAF 绕过奖励模型训练指南.md](./WAF 绕过奖励模型训练指南.md)

---

**作者**: WAF-LLM-Adversarial-Workbench  
**日期**: 2026-01-19  
**版本**: 1.0
