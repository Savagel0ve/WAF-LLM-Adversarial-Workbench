# GPTFuzzer Stage 3: Reinforcement Learning (强化学习)

本目录包含GPTFuzzer框架的第三阶段（强化学习）的完整实现。

---

## 📂 文件结构

```
├── train/
│   ├── train_rl.py              # 主训练脚本 (PPO)
│   ├── test_rl_model.py         # 测试训练好的模型
│   ├── generate_payloads.py     # 批量生成载荷
│   └── evaluate_rl.py           # 评估WAF绕过率
├── train_rl_sqli.ps1            # PowerShell启动脚本
├── QUICKSTART_RL.md             # 快速入门指南
└── README_RL.md                 # 本文件
```

---

## 🎯 什么是强化学习阶段？

在完成预训练（Stage 1）和奖励模型训练（Stage 2）后，强化学习阶段使用**PPO算法**微调预训练模型，使其能够生成**绕过WAF**的载荷。

### 核心思路

```
预训练模型 (会写SQL语法)
    ↓
强化学习 (PPO)
    ↓
WAF绕过模型 (会写绕过WAF的SQL注入)
```

### 关键组件

1. **Policy Network (策略网络)**: 从预训练模型初始化，负责生成载荷
2. **Reference Model (参考模型)**: 冻结的预训练模型，用于计算KL散度
3. **Reward Model (奖励模型)**: 训练好的分类器，评估载荷绕过WAF的概率
4. **PPO算法**: 在最大化奖励和保持语法正确性之间平衡

---

## 🚀 快速开始

### 前置条件

确保已完成：
- ✅ Stage 1: 预训练 (`./models/pretrain_sqli_gpt2_small/`)
- ✅ Stage 2: 奖励模型 (`./models/reward_sqli/final_reward_model/`)

### 一键训练

```powershell
# 使用PowerShell脚本（推荐）
.\train_rl_sqli.ps1
```

或使用Python直接运行：

```bash
python train/train_rl.py \
    --pretrained_model ./models/pretrain_sqli_gpt2_small \
    --reward_model ./models/reward_sqli/final_reward_model \
    --output_dir ./models/rl_sqli_gpt2 \
    --total_episodes 20
```

### 训练时间

- **RTX 4070 (8GB)**: 约 2-4 小时 (20轮)
- **RTX 4090 (24GB)**: 约 1-2 小时

---

## 📊 训练参数详解

### 论文推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `learning_rate` | **1.4e-5** | 非常小，避免忘记语法 |
| `batch_size` | **256** | 每轮生成256个样本 |
| `mini_batch_size` | **16** | PPO更新的mini batch |
| `init_kl_coef` (β) | **0.2** | 🔥 **最关键参数** |
| `ppo_epochs` | **4** | PPO内部更新次数 |
| `total_episodes` | **20** | 总训练轮数 |

### 奖励函数

```
R_total = R_WAF - β · KL(π_θ, ρ)
```

- **R_WAF**: 奖励模型输出的绕过概率 (0-1)
- **KL(π_θ, ρ)**: 当前策略与预训练模型的KL散度
- **β = 0.2**: 平衡绕过率和语法正确性

---

## 🧪 测试和评估

### 1. 快速测试生成效果

```bash
python train/test_rl_model.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --num_samples 50
```

输出：
```
📊 评估 50 个载荷...
  - 总数: 50
  - 唯一: 48 (96.0%)
  - 平均长度: 45.2 字符
  - 有效: 46 (92.0%)

📄 载荷样例:
[1] ' union select 1,2,3--
[2] 1' and 1=1--
...
```

### 2. 批量生成载荷

```bash
python train/generate_payloads.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --num_samples 1000 \
    --output_file ./generated_payloads.txt \
    --deduplicate
```

### 3. 评估WAF绕过率

```bash
python train/evaluate_rl.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --waf_url http://localhost:8001 \
    --num_samples 100 \
    --output_file ./evaluation_results.json
```

输出：
```
📊 评估结果
生成统计:
  - 总生成: 100
  - 唯一: 95 (95.0%)
  - 语法有效: 90 (94.7%)

WAF测试:
  - 测试总数: 90
  - 绕过: 45 (50.0%)
  - 被阻止: 43 (47.8%)
  - 错误: 2

🎯 最终绕过率: 50.00%
```

### 4. 功能性验证（DVWA）

```bash
python train/evaluate_rl.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --waf_url http://localhost:8082 \
    --num_samples 100 \
    --functional_verify \
    --dvwa_login \
    --dvwa_username admin \
    --dvwa_password password \
    --fv_url http://localhost:8081/vulnerabilities/sqli/ \
    --fv_param id \
    --fv_method get \
    --fv_success_regex "First name|Surname" \
    --output_file ./evaluation_results.json
```

若不提供 `--fv_url`，会输出人工检查样本到 `results/functional_verification_samples.json`。

---

## 📈 监控训练

### 关键指标

训练过程中监控以下指标：

1. **平均奖励 (Mean Reward)**
   - 初始: 0.01 ~ 0.1
   - 目标: 0.3 ~ 0.5+
   - 应该逐渐上升

2. **KL散度 (KL Divergence)**
   - 目标: 保持在 0.1 左右
   - 太大: 模型偏离语法太远
   - 太小: 模型没有充分学习

3. **生成质量**
   - 语法有效率: 应该 > 90%
   - 多样性: 唯一率应该 > 80%

---

## 💡 常见问题

### Q1: 显存不足 (OOM)

**解决方案**:

```bash
# 降低batch size
python train/train_rl.py --batch_size 128 --mini_batch_size 8

# 或减小生成长度
python train/train_rl.py --max_new_tokens 64
```

### Q2: 平均奖励一直很低

**可能原因**:
1. 奖励模型质量差 → 重新训练奖励模型
2. KL系数太大 → 降低 `--init_kl_coef` 到 0.1
3. 预训练不充分 → 使用更多数据重新预训练

### Q3: 生成的载荷语法错误率高

**解决方案**:

```bash
# 增大KL系数（保持更接近预训练）
python train/train_rl.py --init_kl_coef 0.3
```

### Q4: 生成结果都是重复的

**解决方案**:

```bash
# 测试时增加温度
python train/test_rl_model.py --temperature 1.5 --top_k 100
```

---

## 🎓 技术细节

### PPO算法

PPO（Proximal Policy Optimization）是一种策略梯度算法，通过限制每次更新的步长来保证训练稳定性。

关键特性：
- **Clipped Objective**: 防止策略更新过大
- **Value Function**: 估计状态价值，减少方差
- **KL Penalty**: 约束策略不要偏离太远

### 与预训练的区别

| 阶段 | 目标 | 训练信号 | 结果 |
|------|------|----------|------|
| **预训练** | 学习语法 | 下一个token | 生成语法正确的载荷 |
| **强化学习** | 绕过WAF | WAF响应 | 生成能绕过WAF的载荷 |

---

## 📚 相关文档

- [QUICKSTART_RL.md](./QUICKSTART_RL.md) - 详细的快速入门指南
- [GPTFuzzer 强化学习阶段复现指南.md](./GPTFuzzer%20强化学习阶段复现指南.md) - 实现细节
- [GPTFuzzer 论文细节总结.md](./GPTFuzzer%20论文细节总结.md) - 论文解读

---

## 🔗 引用

如果你使用本实现，请引用原论文：

```bibtex
@inproceedings{gptfuzzer2023,
  title={GPTFuzzer: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts},
  author={Yu, Jiahao and others},
  booktitle={Proceedings of the Network and Distributed System Security Symposium},
  year={2024}
}
```

---

## ⚠️ 免责声明

本工具仅用于安全研究和教育目的。请勿用于非法攻击。使用者需遵守当地法律法规。

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**祝训练顺利！🚀**
