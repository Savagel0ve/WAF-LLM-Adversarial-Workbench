# GPTFuzzer Stage 3: Reinforcement Learning 快速入门

本文档介绍如何使用强化学习（PPO算法）训练WAF绕过模型。

---

## 📋 前置条件

在开始RL训练之前，请确保已完成：

✅ **Stage 1: 预训练** - 完成语法学习  
   - 模型位置: `./models/pretrain_sqli_gpt2_small/`
   - 训练脚本: `train_sqli.ps1`

✅ **Stage 2: 奖励模型** - 训练WAF分类器  
   - 模型位置: `./models/reward_sqli/final_reward_model/`
   - 训练脚本: `train_reward_sqli.ps1`

✅ **环境配置**  
   - Python 3.8+
   - PyTorch 2.0+
   - CUDA (推荐，否则训练非常慢)
   - 依赖包: `pip install -r train/requirements_train.txt`

---

## 🚀 快速开始

### 方法1: 使用PowerShell脚本 (推荐)

```powershell
# 直接运行，自动检查环境和模型
.\train_rl_sqli.ps1
```

脚本会自动：
- ✅ 检查预训练模型和奖励模型是否存在
- ✅ 激活虚拟环境
- ✅ 配置训练参数（使用论文推荐值）
- ✅ 开始训练

### 方法2: 直接使用Python

```bash
python train/train_rl.py \
    --pretrained_model ./models/pretrain_sqli_gpt2_small \
    --reward_model ./models/reward_sqli/final_reward_model \
    --output_dir ./models/rl_sqli_gpt2 \
    --total_episodes 20 \
    --batch_size 256 \
    --mini_batch_size 16 \
    --init_kl_coef 0.2 \
    --learning_rate 1.4e-5
```

---

## ⚙️ 核心参数说明

### 必需参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrained_model` | Stage 1的预训练模型路径 | `./models/pretrain_sqli_gpt2_small` |
| `--reward_model` | Stage 2的奖励模型路径 | `./models/reward_sqli/final_reward_model` |
| `--output_dir` | 输出目录 | `./models/rl_sqli_gpt2` |

### PPO超参数 (来自论文)

| 参数 | 论文推荐值 | 说明 |
|------|-----------|------|
| `--learning_rate` | **1.4e-5** | 学习率，非常小以避免忘记语法 |
| `--batch_size` | **256** | 每轮生成的样本数 |
| `--mini_batch_size` | **16** | PPO更新的mini batch (显存优化) |
| `--init_kl_coef` (β) | **0.2** | 🔥 **最关键参数**，控制KL散度惩罚 |
| `--ppo_epochs` | **4** | PPO内部更新轮数 |

### 训练控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total_episodes` | 20 | 总训练轮数 |
| `--max_new_tokens` | 128 | 每次生成的最大token数 |
| `--save_freq` | 5 | 每N轮保存一次检查点 |
| `--temperature` | 1.0 | 生成温度 (0.1-2.0) |
| `--seed` | 42 | 随机种子 |

---

## 📊 训练过程理解

### PPO工作流程

每个训练轮次（Episode）包含以下步骤：

```
1. 🎲 Generate (Rollout)
   使用当前策略网络生成 batch_size 个载荷
   
2. 🎁 Calculate Rewards
   奖励模型评估每个载荷的WAF绕过概率 (0-1)
   
3. 🔄 PPO Update
   根据奖励和KL散度更新策略网络参数
   
4. 💾 Save Checkpoint
   定期保存模型
```

### 奖励函数设计 (论文核心)

```
R_total = R_WAF - β · KL(π_θ, ρ)
```

- **R_WAF**: 奖励模型输出的绕过概率 (0-1)
  - 只在生成结束时给予（最后一个token）
  
- **KL(π_θ, ρ)**: KL散度惩罚
  - 衡量当前策略与原始预训练模型的偏离程度
  - 每个生成步骤都计算
  
- **β**: KL系数（论文推荐0.2）
  - 太小: 模型过度优化奖励，可能生成无效语法
  - 太大: 模型不敢偏离预训练，难以学到绕过技巧

---

## 📈 监控训练进度

### 关键指标

训练过程中会输出以下指标：

```
📊 奖励统计:
   - 平均奖励: 0.1234    # 应该逐渐上升
   - 最大奖励: 0.8901    # 最好的样本
   - 最小奖励: 0.0012    # 最差的样本

🔄 PPO更新:
   - PPO平均分数: 0.1234
   - KL散度: 0.0567       # 应该保持在0.1左右
   - 总损失: 1.234
```

### 收敛标准

- **初始阶段** (Episodes 1-5):
  - 平均奖励: 0.01 ~ 0.1
  - 模型在探索各种变体
  
- **中期** (Episodes 6-15):
  - 平均奖励: 0.1 ~ 0.3
  - 模型找到一些有效的绕过模式
  
- **收敛** (Episodes 16-20):
  - 平均奖励: 0.3 ~ 0.5+ (取决于WAF难度)
  - 奖励稳定，不再大幅波动

---

## 🧪 测试训练好的模型

训练完成后，模型保存在 `./models/rl_sqli_gpt2/final_model/`

### 1. 快速测试生成效果

```bash
python train/test_rl_model.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --num_samples 50
```

输出示例：
```
🎲 生成 50 个载荷...
   [1/50] ' union select 1,2,3--
   [10/50] 1' and 1=1--
   ...

📊 评估 50 个载荷...
  - 总数: 50
  - 唯一: 48 (96.0%)
  - 重复: 2
  - 平均长度: 45.2 字符
  - 有效: 46 (92.0%)
```

### 2. 生成大量载荷

```bash
python train/test_rl_model.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --num_samples 1000 \
    --output_file ./generated_payloads.txt
```

### 3. 调整生成参数

```bash
# 更随机的生成 (探索性更强)
python train/test_rl_model.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --temperature 1.5 \
    --top_k 100 \
    --num_samples 50

# 更确定性的生成 (更接近训练分布)
python train/test_rl_model.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_samples 50
```

---

## 📊 评估与功能验证

### 1. 评估WAF绕过率

```bash
python train/evaluate_rl.py \
    --model_path ./models/rl_sqli_gpt2/final_model \
    --waf_url http://localhost:8082 \
    --num_samples 100 \
    --output_file ./evaluation_results.json
```

### 2. 功能性验证（DVWA）

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

不传 `--fv_url` 会自动输出 100 个绕过样本供人工检查：
`results/functional_verification_samples.json`。

---

## 💡 常见问题

### Q1: 训练过程中显存不足 (OOM)

**解决方案**:

1. 减小batch size:
```bash
python train/train_rl.py --batch_size 128 --mini_batch_size 8
```

2. 减小生成长度:
```bash
python train/train_rl.py --max_new_tokens 64
```

3. 使用8-bit量化 (需要修改代码):
```python
# 在 train_rl.py 中，加载模型时:
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_path,
    load_in_8bit=True
)
```

### Q2: 平均奖励一直很低 (< 0.05)

**可能原因**:

1. **奖励模型质量差**
   - 检查Stage 2的测试准确率，应该 > 90%
   - 解决: 用更多数据重新训练奖励模型

2. **KL系数太大**
   - 模型不敢偏离预训练分布
   - 解决: 降低 `--init_kl_coef` 到 0.1

3. **预训练不充分**
   - Stage 1的预训练数据太少
   - 解决: 使用更多数据重新预训练

### Q3: 生成的载荷都是重复的

**解决方案**:

1. 增加生成随机性:
```bash
python train/test_rl_model.py --temperature 1.5 --top_k 100
```

2. 可能是模型过拟合了，尝试：
   - 减少训练轮数
   - 增大KL系数

### Q4: 生成的载荷语法错误率高

**解决方案**:

1. 增大KL系数（保持更接近预训练分布）:
```bash
python train/train_rl.py --init_kl_coef 0.3
```

2. 检查预训练模型质量（Stage 1）

### Q5: 训练速度太慢

**优化方案**:

1. 使用更小的batch size但更多轮数:
```bash
python train/train_rl.py --batch_size 128 --total_episodes 40
```

2. 减少生成长度:
```bash
python train/train_rl.py --max_new_tokens 64
```

3. 使用混合精度训练 (已默认启用):
```python
config.use_fp16 = True  # 在代码中已启用
```

---

## 📁 输出文件结构

训练完成后，输出目录结构如下：

```
models/rl_sqli_gpt2/
├── checkpoint-5/              # 第5轮检查点
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── checkpoint-10/             # 第10轮检查点
├── checkpoint-15/             # 第15轮检查点
├── checkpoint-20/             # 第20轮检查点
├── final_model/               # 🎯 最终模型 (使用这个)
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── rl_config.json
├── training_stats.json        # 训练统计
└── rl_config.json             # 训练配置
```

---

## 🎯 下一步

完成RL训练后，你可以：

1. **集成到Web应用**
   - 将模型集成到后端API
   - 实时生成WAF绕过载荷

2. **持续优化**
   - 收集模型生成的成功案例
   - 添加到训练数据，重新训练

3. **扩展到其他攻击类型**
   - 使用相同方法训练XSS、RCE模型
   - 只需替换预训练模型和奖励模型

4. **评估和基准测试**
   - 在真实WAF上测试绕过率
   - 与传统模糊测试工具对比

---

## 📚 参考资料

- **论文**: GPTFuzzer: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts
- **PPO算法**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **TRL库**: [Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- **相关文档**:
  - [GPTFuzzer 强化学习阶段复现指南.md](./GPTFuzzer%20强化学习阶段复现指南.md)
  - [GPTFuzzer 论文细节总结.md](./GPTFuzzer%20论文细节总结.md)

---

## ⚠️ 重要提示

1. **显存需求**: 
   - 最低8GB (使用优化参数)
   - 推荐12GB+ (可用论文原始参数)

2. **训练时间**:
   - RTX 4070: 约2-4小时 (20轮，batch_size=256)
   - CPU: 不推荐 (太慢)

3. **伦理使用**:
   - 仅用于安全测试和研究
   - 不得用于非法攻击
   - 遵守当地法律法规

---

**祝训练顺利！如有问题，请查看日志输出或提issue。**
