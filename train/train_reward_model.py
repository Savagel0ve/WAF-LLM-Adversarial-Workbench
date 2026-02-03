"""
奖励模型训练 - 支持 Qwen2.5-Coder 和其他现代 LLM

根据GPTFuzzer论文:
- 基于预训练模型（Qwen2.5-Coder 或其他）
- 训练序列分类器预测WAF绕过概率
- 使用BCEWithLogitsLoss
- 输出 r(τ) ∈ [0, 1]
"""
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RewardModelConfig:
    """奖励模型配置"""
    
    def __init__(self, args):
        self.pretrained_model_path = args.pretrained_model_path
        self.data_path = args.data_path
        self.output_dir = args.output_dir
        
        # 论文超参数
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.warmup_ratio = args.warmup_ratio
        self.weight_decay = args.weight_decay
        
        # 精度配置
        self.seed = args.seed
        self.fp16 = getattr(args, 'fp16', False)
        self.bf16 = getattr(args, 'bf16', True)  # 默认使用 bf16
        self.early_stopping_patience = args.early_stopping_patience
        
        logger.info("奖励模型配置:")
        for key, value in self.__dict__.items():
            logger.info(f"  {key}: {value}")


def load_and_process_data(config: RewardModelConfig, tokenizer):
    """
    加载CSV数据并转换为HF Dataset格式
    
    支持两种输入格式:
    1. 单个CSV文件 - 自动划分train/val/test
    2. 多个CSV文件 - train.csv, val.csv, test.csv
    """
    data_path = Path(config.data_path)
    
    if data_path.is_file():
        # 单个文件 - 自动划分
        logger.info(f"加载数据文件: {data_path}")
        df = pd.read_csv(data_path)
        
        # 随机划分
        df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
        
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
    else:
        # 多个文件
        logger.info(f"从目录加载数据: {data_path}")
        
        # 查找文件
        train_file = None
        val_file = None
        test_file = None
        
        for file in data_path.glob("*.csv"):
            if "train" in file.name:
                train_file = file
            elif "val" in file.name:
                val_file = file
            elif "test" in file.name:
                test_file = file
        
        if not train_file:
            raise ValueError(f"找不到训练文件在 {data_path}")
        
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file) if val_file else None
        test_df = pd.read_csv(test_file) if test_file else None
    
    logger.info(f"数据统计: Train={len(train_df)}, Val={len(val_df) if val_df is not None else 0}, Test={len(test_df) if test_df is not None else 0}")
    
    # 检查列
    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("CSV文件必须包含 'text' 和 'label' 列")
    
    # 数据统计
    logger.info(f"训练集正样本比例: {train_df['label'].mean():.2%}")
    if val_df is not None:
        logger.info(f"验证集正样本比例: {val_df['label'].mean():.2%}")
    if test_df is not None:
        logger.info(f"测试集正样本比例: {test_df['label'].mean():.2%}")
    
    # 转换为Dataset
    datasets_dict = {"train": Dataset.from_pandas(train_df)}
    if val_df is not None:
        datasets_dict["validation"] = Dataset.from_pandas(val_df)
    if test_df is not None:
        datasets_dict["test"] = Dataset.from_pandas(test_df)
    
    raw_datasets = DatasetDict(datasets_dict)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )
    
    logger.info("Tokenizing数据集...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing"
    )
    
    return tokenized_datasets


def compute_metrics(eval_pred):
    """
    计算评估指标
    
    指标包括:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - AUC-ROC
    """
    logits, labels = eval_pred
    
    # 转换为概率
    probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
    predictions = (probs > 0.5).astype(int)
    
    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    # AUC
    try:
        auc = roc_auc_score(labels, probs)
    except Exception as e:
        logger.warning(f"无法计算AUC: {e}")
        auc = 0.0
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


class RewardTrainer(Trainer):
    """Custom Trainer - using BCEWithLogitsLoss"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # BCEWithLogitsLoss - numerically stable version
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.float().view(-1))
        
        return (loss, outputs) if return_outputs else loss


def train(args):
    """训练奖励模型 - 支持 Qwen2.5-Coder 和其他现代 LLM"""
    config = RewardModelConfig(args)
    set_seed(config.seed)
    
    logger.info("="*60)
    logger.info("🎯 开始训练奖励模型")
    logger.info("="*60)
    
    # 加载 tokenizer
    logger.info(f"📝 加载 tokenizer: {config.pretrained_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path,
        trust_remote_code=True,
        padding_side="right",  # 分类任务使用 right padding
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"   设置 pad_token 为 eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("   添加了新的 pad_token: [PAD]")
    
    # 加载并处理数据
    tokenized_datasets = load_and_process_data(config, tokenizer)
    
    # 确定精度
    torch_dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    
    # 加载模型
    logger.info(f"🤖 加载预训练模型: {config.pretrained_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=1,  # 二分类，使用BCEWithLogitsLoss
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    # 确保 pad_token_id 设置正确
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 如果添加了新的特殊token，需要调整embedding
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   总参数: {total_params / 1e9:.2f}B ({total_params / 1e6:.1f}M)")
    logger.info(f"   可训练: {trainable_params / 1e9:.2f}B ({trainable_params / 1e6:.1f}M)")
    
    # 设备信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  使用设备: {device}")
    if device == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        
        # 日志和保存
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=10,
        logging_first_step=True,
        
        # 评估
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # 精度配置 - 优先使用 bf16
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=config.fp16 and not config.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=4,
        
        # 报告
        report_to="tensorboard",
        
        # 其他
        seed=config.seed,
        save_total_limit=3,
    )
    
    # Callbacks
    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        )
    
    # 创建Trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # 训练
    logger.info("\n开始训练...")
    train_result = trainer.train()
    
    # 保存最终模型
    final_model_path = Path(config.output_dir) / "final_reward_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✅ 模型保存至: {final_model_path}")
    
    # 训练指标
    logger.info("\n" + "="*60)
    logger.info("训练结果:")
    for key, value in train_result.metrics.items():
        logger.info(f"  {key}: {value}")
    
    # 在测试集上评估
    if "test" in tokenized_datasets:
        logger.info("\n" + "="*60)
        logger.info("在测试集上评估...")
        test_results = trainer.predict(tokenized_datasets["test"])
        
        logger.info("测试集结果:")
        for key, value in test_results.metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 保存测试结果
        test_results_path = Path(config.output_dir) / "test_results.json"
        import json
        with open(test_results_path, 'w') as f:
            json.dump(test_results.metrics, f, indent=2)
    
    logger.info("\n✅ 训练完成!")
    logger.info(f"查看训练日志: tensorboard --logdir {config.output_dir}/logs")


def main():
    parser = argparse.ArgumentParser(
        description="训练WAF绕过奖励模型 - 支持 Qwen2.5-Coder 和其他现代 LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用 Qwen2.5-Coder 预训练模型
  python train_reward_model.py \\
      --pretrained_model_path ./models/pretrain_sqli_qwen2_5_coder_1_5b \\
      --data_path ./data/labeled \\
      --output_dir ./models/reward_sqli_qwen

  # 使用 BF16 精度 (推荐)
  python train_reward_model.py \\
      --pretrained_model_path ./models/pretrain_sqli_qwen2_5_coder_1_5b \\
      --data_path ./data/labeled \\
      --bf16
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="预训练模型路径 (Qwen2.5-Coder 或其他)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="标记数据路径 (CSV文件或目录)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/reward_model",
        help="输出目录"
    )
    
    # 超参数 (论文默认值)
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度 (默认: 256)")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小 (默认: 16，Qwen 1.5B 推荐)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    # 精度配置
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--bf16", action="store_true", default=True, help="使用 BF16 精度 (默认开启)")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 精度")
    parser.add_argument("--no-bf16", action="store_true", help="禁用 BF16")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="早停patience (0=不使用)")
    
    args = parser.parse_args()
    
    # 处理精度参数
    if args.no_bf16:
        args.bf16 = False
    if args.fp16:
        args.bf16 = False
    
    train(args)


if __name__ == "__main__":
    main()
