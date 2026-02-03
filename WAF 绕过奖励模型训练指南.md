# WAF 绕过奖励模型训练指南

这是一个基于论文《Generative Pre-Trained Transformer-Based Reinforcement Learning for Testing Web Application Firewalls》定制的奖励模型训练复现指南。

这份指南帮助你利用预训练模型，训练一个判断 Payload 是否绕过 WAF 的判别器。

---

## 1. 阶段概述与目标

奖励模型训练是连接“预训练”与“强化学习”的关键桥梁。

- **输入**：经过预训练的 GPT-2 模型权重。
- **架构**：移除预训练模型的语言建模头，替换为奖励预测头（线性层 + Sigmoid 激活）。
- **目标**：输入攻击载荷序列，输出标量 `r(τ) ∈ [0, 1]`，代表绕过概率。
- **损失函数**：二元交叉熵损失（Binary Cross-Entropy Loss）。

---

## 2. 数据集准备 (Data Preparation)

在开始训练代码之前，需要构建标记数据集。

### 2.1 数据采样与打标

根据论文描述，不需要海量数据，只需少量高质量交互数据：

- **采样来源**：从收集或生成的攻击载荷池中随机采样。
- **采样数量**：
  - **SQLi**：4,000 条
  - **XSS / RCE**：2,000 条
- **打标方式**：将采样出的 Payload 发送给目标 WAF（如 ModSecurity 或 Naxsi）。
  - **Label 1 (Bypassing)**：WAF 未拦截（如返回 200 OK）
  - **Label 0 (Blocked)**：WAF 拦截（如返回 403 Forbidden）

### 2.2 数据集划分

将打标后的数据集按以下比例划分：

- **训练集 (Training)**：70%
- **验证集 (Validation)**：15%
- **测试集 (Testing)**：15%

### 2.3 建议数据格式 (CSV)

```csv
text,label
"0' OR 1=1--",1
"<script>alert(1)</script>",0
...
```

---

## 3. 核心超参数配置 (Hyperparameters)

论文中给出的奖励模型训练参数如下：

| 参数项 | 论文设定值 | 说明 |
| :---- | :---- | :---- |
| **Epochs** | 4 | 训练轮数较少，避免过拟合 |
| **Batch Size** | 32 | RTX 4070 可轻松处理 |
| **Learning Rate** | 2e-5 | 比预训练更小的学习率 |
| **LR Schedule** | Linear Warmup + Decay | 前 10% 步数线性预热，之后线性衰减至 0 |
| **Optimizer** | Adam | 标准优化器 |

---

## 4. 复现代码实现 (train_reward_model.py)

此代码基于 HuggingFace Transformers，适配 RTX 4070。它加载预训练模型，并将其转换为序列分类模型。

```python
import torch
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


class RewardConfig:
    def __init__(self, args):
        self.pretrained_model_path = args.pretrained_model_path
        self.data_path = args.data_path
        self.output_dir = args.output_dir
        self.max_length = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.epochs = 4
        self.warmup_ratio = 0.1
        self.seed = 42


def load_and_process_data(config, tokenizer):
    """加载 CSV 数据并转换为 HF Dataset 格式。"""
    df = pd.read_csv(config.data_path)

    train_df, val_test_df = np.split(
        df.sample(frac=1, random_state=config.seed),
        [int(0.7 * len(df))]
    )
    val_df, test_df = np.split(
        val_test_df,
        [int(0.5 * len(val_test_df))]
    )

    print(f"数据划分: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    return tokenized_datasets


def compute_metrics(eval_pred):
    """计算 F1、AUC 等指标。"""
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
    predictions = (probs > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


def train(args):
    config = RewardConfig(args)
    set_seed(config.seed)

    print(f"正在加载预训练模型: {config.pretrained_model_path} ...")

    tokenizer = GPT2Tokenizer.from_pretrained(config.pretrained_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = load_and_process_data(config, tokenizer)

    model = GPT2ForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=config.warmup_ratio,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="tensorboard",
    )

    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    print("开始训练奖励模型...")
    trainer.train()

    trainer.save_model(f"{config.output_dir}/final_reward_model")
    print(f"模型已保存至 {config.output_dir}/final_reward_model")

    print("在测试集上评估...")
    test_results = trainer.predict(tokenized_datasets["test"])
    print(test_results.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./reward_model_output")
    args = parser.parse_args()
    train(args)
```

---

## 5. 复现关键点与验证 (Checklist)

在运行上述脚本时，请注意以下与论文细节的对应：

1. **输入来源**：`--pretrained_model_path` 必须指向上一阶段预训练模型目录，而不是原始 `gpt2`。
2. **损失函数**：使用 `BCEWithLogitsLoss`，等价于先 Sigmoid 再算 BCE，数值更稳定。
3. **评估指标**：观察 F1-Score 和 AUC。论文中 SQLi AUC > 99%、XSS AUC > 98%。
4. **硬件适配**：脚本开启 `fp16=True`，显著降低显存占用并加速训练。

---

## 6. 下一步

训练完成后，你将得到一个目录（如 `reward_model_output/final_reward_model`），该路径用于 PPO 强化学习阶段的 `reward_model` 输入。