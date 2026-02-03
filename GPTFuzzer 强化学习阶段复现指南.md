# GPTFuzzer 强化学习阶段复现指南

完成 **奖励模型（Reward Model）** 训练后，下一阶段是 **强化学习（Reinforcement Learning, RL）**。这是 GPTFuzzer 的核心阶段，模型将从“生成语法正确的 payload”演进到“生成可绕过 WAF 的 payload”。

---

## 1. 阶段逻辑与目标

### 输入

1. **策略网络（Policy Network）**：由预训练语言模型（Stage 1）初始化
2. **参考模型（Reference Model）**：冻结参数的预训练模型（用于计算 KL 散度）
3. **奖励模型（Reward Model）**：Stage 2 训练好的分类器（提供奖励信号）

### 算法

**PPO（Proximal Policy Optimization）**

### 目标

微调策略网络，使其生成的 payload：
- 得分高（可绕过 WAF）
- 不偏离原始语法分布太多（由 KL 散度约束）

---

## 2. 核心公式（实现关键）

在写代码前必须理解奖励函数设计，这是论文中最重要的细节：

```
R_total = R_WAF - β · KL(π_θ, ρ)
```

其中：

- **R_WAF**：奖励模型给出的概率（0~1）。**仅在生成结束（最后一个 token）给出**
- **KL(π_θ, ρ)**：当前策略 π 与原始预训练模型 ρ 的 KL 散度。**每一步生成都要计算并惩罚**
- **β（Beta）**：KL 惩罚系数。论文最优值：**0.2**

---

## 3. 超参数（来自论文）

论文 Section III-H.3 给出的参数：

| 参数 | 论文数值 | 说明 |
|------|----------|------|
| **Algorithm** | PPO | 策略梯度算法 |
| **Learning Rate** | 1.4e-5 | 极小，防止遗忘语法 |
| **Batch Size** | 256 | PPO 更新批大小 |
| **Clip Epsilon (ε)** | 0.2 | PPO 剪切阈值 |
| **KL Coefficient (β)** | 0.2 | 防止模型崩塌的关键参数 |
| **Epochs** | ~5-20 | 通常 5 轮即可得到较好结果 |

---

## 4. 实现代码结构（`train_rl.py`）

建议使用 `trl`（Transformer Reinforcement Learning）库来实现 PPO。该库由 HuggingFace 维护，对 GPT-2 和 PPO 支持完善。

```python
import torch
from transformers import GPT2Tokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import pipeline

# === 1. 配置 ===
config = PPOConfig(
    model_name="./models/pretrain_xss_gpt2",  # Stage 1 预训练模型路径
    learning_rate=1.4e-5,                     # 论文参数
    batch_size=256,                           # 论文参数
    mini_batch_size=16,                       # 内存优化：256/16 = 16 次累积
    ppo_epochs=4,                             # 每次采样的训练轮数
    init_kl_coef=0.2,                         # 论文 beta 参数
    adap_kl_ctrl=False,                       # 论文使用固定 beta，关闭自适应
)

# === 2. 加载模型 ===
# Policy Network - 带 Value Head
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Reference Model - 用于 KL（冻结参数）
ref_model = create_reference_model(model)

# Reward Model - Stage 2 输出
reward_pipe = pipeline(
    "text-classification",
    model="./reward_model_output/final_reward_model",  # Stage 2 模型路径
    device=0,
    tokenizer=tokenizer
)

# === 3. PPO Trainer ===
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# === 4. 训练循环（伪代码）===
# 训练 20 个 step
for step in range(20):
    
    # --- A. 生成（Rollout）---
    # 随机生成 batch_size 个 query（通常是 <start> token）
    query_tensors = [torch.tensor([tokenizer.bos_token_id]) for _ in range(config.batch_size)]
    
    # 用当前策略网络生成响应
    response_tensors = ppo_trainer.generate(
        query_tensors,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id
    )
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
    # --- B. 计算 Reward ---
    # 用奖励模型打分
    # 注意：论文中的 reward model 输出 [0,1] 概率，直接作为奖励
    pipe_outputs = reward_pipe(batch['response'], return_all_scores=True)
    # 取 Label '1'（绕过）概率
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
    # --- C. PPO 更新 ---
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    print(f"Step {step}: Mean Reward = {stats['ppo/mean_scores']:.4f}")

# === 5. 保存最终模型 ===
model.save_pretrained("./rl_final_model")
tokenizer.save_pretrained("./rl_final_model")
```

---

## 5. 实现注意事项

### 1. Value Head 初始化

使用 `AutoModelForCausalLMWithValueHead`，因为 PPO 需要 Value Function 来估计当前状态价值。GPTFuzzer 论文提到 Policy 和 Value 网络共享 Transformer 主体，这也是 trl 的默认实现。

### 2. 显存压力

RL 阶段需要同时加载 **3 个模型**：

1. **Policy Model**（训练，需要梯度）
2. **Reference Model**（冻结）
3. **Reward Model**（冻结）

**RTX 4070（8GB/12GB）** 可能会 OOM。

**优化建议**：
- 使用 `mini_batch_size` 拆分 256 的大 batch
- 将 Reference Model 和 Reward Model 量化到 8-bit（`load_in_8bit=True`）
- 把 Reward Model 放到 CPU（更慢但可行）

### 3. 收敛指标

关注 **Mean Reward**：
- 初期：0.01~0.1（取决于预训练模型的初始能力）
- 训练后：应逐步上升并稳定
  - 0.9+（攻击类型较易绕过）
  - 0.1~0.2（难度高，如 XSS）

---

## 6. 下一步

完成 RL 训练后，你将得到：

1. **训练好的策略模型**：可生成绕过 WAF 的 payload
2. **完整的 GPTFuzzer 系统**：可用于测试与评估

### 测试模型

```python
# Load trained model
model = AutoModelForCausalLM.from_pretrained("./rl_final_model")
tokenizer = GPT2Tokenizer.from_pretrained("./rl_final_model")

# Generate payloads
input_ids = tokenizer.encode("<start>", return_tensors="pt")
outputs = model.generate(input_ids, max_length=128, num_return_sequences=10)

# Decode and test
for output in outputs:
    payload = tokenizer.decode(output, skip_special_tokens=True)
    print(payload)
```

---

## 7. 参考资料

- **论文**："GPTFuzzer: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts"
- **trl 库**：https://github.com/huggingface/trl
- **PPO 算法**：https://arxiv.org/abs/1707.06347

---

**祝你 RL 实现顺利！这是最后一步，完成后就拥有完整可用的 GPTFuzzer 系统。**
