"""
预训练脚本 - 支持 Qwen2.5-Coder 和其他现代 LLM
针对 RTX 4070 8GB 优化

支持的模型:
- Qwen2.5-Coder-0.5B/1.5B/3B (推荐)
- Qwen2.5-0.5B/1.5B
- DeepSeek-Coder-1.3B
- Phi-3-mini
- GPT-2 (兼容旧版)
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig,
)
from config import (
    ModelConfig, 
    TrainingConfig, 
    GPUConfig,
    MODEL_PRESETS,
    DEFAULT_MODEL,
    get_quantization_config,
)


class PayloadDataset:
    """Payload数据集加载器"""
    
    def __init__(self, data_dir="data/processed", attack_type="xss"):
        self.data_dir = Path(data_dir) / attack_type
        self.attack_type = attack_type
    
    def load(self):
        """加载数据集"""
        data_files = {}
        
        for split, filename in [("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")]:
            path = self.data_dir / filename
            if path.exists():
                data_files[split] = str(path)
            else:
                print(f"警告: {path} 不存在，跳过 {split} 集")
        
        if not data_files:
            raise FileNotFoundError(f"未找到 {self.attack_type} 数据集文件")
        
        dataset = load_dataset('text', data_files=data_files)
        return dataset


def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenize数据集"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer(model_name: str, gpu_config: GPUConfig, model_config: ModelConfig):
    """
    设置模型和tokenizer - 支持多种现代LLM
    
    Args:
        model_name: 模型名称或HuggingFace路径
        gpu_config: GPU配置
        model_config: 模型配置
    """
    print(f"\n🤖 加载模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="left",  # Causal LM 推荐 left padding
    )
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 获取量化配置
    quantization_config = get_quantization_config(gpu_config)
    
    # 确定精度
    if quantization_config:
        torch_dtype = None  # 量化时自动处理
    elif gpu_config.use_bf16:
        torch_dtype = torch.bfloat16
    elif gpu_config.use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # 加载模型 - 显式指定 CUDA 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {
        "trust_remote_code": model_config.trust_remote_code,
    }
    
    # 只有在使用量化时才用 device_map="auto"
    if gpu_config.load_in_4bit or gpu_config.load_in_8bit:
        model_kwargs["device_map"] = "auto"
    # 否则不设置 device_map，让 Trainer 处理设备分配
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        print(f"   使用量化: {'4-bit' if gpu_config.load_in_4bit else '8-bit'}")
    elif torch_dtype:
        model_kwargs["torch_dtype"] = torch_dtype
        print(f"   精度: {torch_dtype}")
    
    # Flash Attention 2 支持 - 检查是否可用
    use_flash_attn = False
    if gpu_config.use_flash_attention:
        try:
            import flash_attn
            use_flash_attn = True
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("   Flash Attention 2: enabled")
        except ImportError:
            print("   Flash Attention 2: not installed, using default attention")
    
    # 加载模型，如果 Flash Attention 失败则回退
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ImportError as e:
        if "flash_attn" in str(e) or "flash_attention" in str(e).lower():
            print("   Flash Attention 2: failed to load, falling back to default")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise
    
    # 显式移动模型到 GPU（如果没有使用 device_map）
    if "device_map" not in model_kwargs and torch.cuda.is_available():
        model = model.to(device)
        print(f"   设备: {device}")
    
    # 调整 pad token id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 如果添加了新的特殊token，需要调整embedding
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数: {total_params / 1e9:.2f}B ({total_params / 1e6:.1f}M)")
    print(f"   可训练: {trainable_params / 1e9:.2f}B ({trainable_params / 1e6:.1f}M)")
    
    return model, tokenizer


def train(args):
    """训练函数 - 支持 Qwen2.5-Coder 和其他现代 LLM"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 根据预设或自定义参数加载配置
    if args.model_preset and args.model_preset in MODEL_PRESETS:
        preset = MODEL_PRESETS[args.model_preset]
        model_name = preset["model_name"]
        max_length = args.max_length or preset.get("max_length", 256)
        batch_size = args.batch_size or preset.get("batch_size", 4)
        gradient_accumulation = args.gradient_accumulation or preset.get("gradient_accumulation", 8)
        use_flash_attention = preset.get("use_flash_attention", True)
        load_in_4bit = preset.get("load_in_4bit", False)
        gradient_checkpointing = preset.get("gradient_checkpointing", False) or args.gradient_checkpointing
    else:
        model_name = args.model_name
        max_length = args.max_length or 256
        batch_size = args.batch_size or 4
        gradient_accumulation = args.gradient_accumulation or 8
        use_flash_attention = args.flash_attention
        load_in_4bit = args.load_in_4bit
        gradient_checkpointing = args.gradient_checkpointing
    
    # 创建配置
    model_config = ModelConfig(
        model_name=model_name,
        max_length=max_length,
        trust_remote_code=True,
    )
    
    gpu_config = GPUConfig(
        use_fp16=not args.bf16,
        use_bf16=args.bf16,
        use_flash_attention=use_flash_attention,
        load_in_4bit=load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    
    print("="*60)
    print(f"🚀 预训练配置 - {args.attack_type.upper()}")
    print("="*60)
    print(f"模型: {model_name}")
    print(f"预设: {args.model_preset or '自定义'}")
    print(f"攻击类型: {args.attack_type}")
    print(f"最大长度: {max_length}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulation: {gradient_accumulation}")
    print(f"等效batch: {batch_size * gradient_accumulation}")
    print(f"学习率: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"精度: {'BF16' if args.bf16 else 'FP16'}")
    print(f"4-bit量化: {load_in_4bit}")
    print(f"Flash Attention: {use_flash_attention}")
    print(f"梯度检查点: {gradient_checkpointing}")
    print(f"优化器: {args.optim}")
    print("="*60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n⚠️  警告: 未检测到GPU，将使用CPU训练")
    
    # 加载数据集
    print(f"\n📂 加载 {args.attack_type} 数据集...")
    dataset_loader = PayloadDataset(args.data_dir, args.attack_type)
    dataset = dataset_loader.load()
    
    print(f"   训练集: {len(dataset['train'])} 条")
    if 'validation' in dataset:
        print(f"   验证集: {len(dataset['validation'])} 条")
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, gpu_config, model_config)
    
    # 验证模型在 GPU 上
    if torch.cuda.is_available():
        # 检查模型参数所在设备
        param_device = next(model.parameters()).device
        print(f"   模型设备: {param_device}")
        if param_device.type != "cuda":
            print("   ⚠️ 模型不在 GPU 上，正在移动...")
            model = model.cuda()
            print(f"   模型已移动到: {next(model.parameters()).device}")
        
        # 显示当前 GPU 内存使用
        print(f"   GPU 内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 启用 gradient checkpointing (节省显存)
    if gpu_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ 启用 gradient checkpointing")
    
    # Tokenize 数据集
    print("\n📝 Tokenizing 数据集...")
    tokenized_dataset = tokenize_dataset(
        dataset, 
        tokenizer, 
        max_length=max_length
    )
    
    # 数据 collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        adam_beta1=train_config.adam_beta1,
        adam_beta2=train_config.adam_beta2,
        adam_epsilon=train_config.adam_epsilon,
        max_grad_norm=train_config.max_grad_norm,
        
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_ratio=train_config.warmup_ratio,
        
        logging_dir=f"{train_config.output_dir}/logs",
        logging_steps=train_config.logging_steps,
        
        save_strategy="steps",
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        
        eval_strategy=train_config.evaluation_strategy if 'validation' in dataset else "no",
        eval_steps=train_config.eval_steps if 'validation' in dataset else None,
        
        load_best_model_at_end=train_config.load_best_model_at_end if 'validation' in dataset else False,
        metric_for_best_model=train_config.metric_for_best_model if 'validation' in dataset else None,
        
        bf16=train_config.bf16,
        fp16=train_config.fp16,
        
        optim=train_config.optim,
        
        dataloader_num_workers=train_config.dataloader_num_workers,
        dataloader_pin_memory=train_config.dataloader_pin_memory,
        
        seed=train_config.seed,
        report_to=["tensorboard"],
        
        gradient_checkpointing=gpu_config.gradient_checkpointing,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # 检测 checkpoint 并恢复
    resume_from_checkpoint = None
    if args.resume:
        checkpoint_dir = Path(train_config.output_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            # 找到最新的 checkpoint（按步数排序）
            def get_step(ckpt):
                try:
                    return int(ckpt.name.split("-")[1])
                except (IndexError, ValueError):
                    return 0
            latest_checkpoint = max(checkpoints, key=get_step)
            resume_from_checkpoint = str(latest_checkpoint)
            print(f"\n🔄 检测到 checkpoint: {latest_checkpoint.name}")
            print(f"   将从 step {get_step(latest_checkpoint)} 恢复训练")
    
    # 开始训练
    print("\n" + "="*60)
    if resume_from_checkpoint:
        print(f"🏃 从 checkpoint 恢复训练...")
    else:
        print("🏃 开始训练...")
    print("="*60)
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存模型
    print("\n💾 保存最终模型...")
    trainer.save_model()
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 评估
    if 'validation' in dataset:
        print("\n📊 评估模型...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        import math
        perplexity = math.exp(eval_metrics['eval_loss']) if eval_metrics['eval_loss'] < 20 else float('inf')
        print(f"\n   验证 Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"   验证 Perplexity: {perplexity:.2f}")
    
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print(f"   模型保存在: {train_config.output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="预训练脚本 - 支持 Qwen2.5-Coder 和其他现代 LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用 Qwen2.5-Coder-1.5B 预设 (推荐)
  python pretrain.py --model-preset qwen2.5-coder-1.5b --attack-type sqli

  # 使用自定义模型
  python pretrain.py --model-name Qwen/Qwen2.5-Coder-0.5B --attack-type xss

  # 使用 4-bit 量化 (大模型)
  python pretrain.py --model-preset qwen2.5-coder-3b --attack-type sqli

可用预设:
  - qwen2.5-coder-0.5b  (推荐快速实验)
  - qwen2.5-coder-1.5b  (推荐，平衡性能和显存)
  - qwen2.5-coder-3b    (需要4-bit量化)
  - deepseek-coder-1.3b (代码专用)
  - phi-3-mini          (需要4-bit量化)
  - gpt2                (兼容旧版)
        """
    )
    
    # 模型选择
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument("--model-preset", type=str, default=DEFAULT_MODEL,
                            choices=list(MODEL_PRESETS.keys()),
                            help=f"模型预设 (默认: {DEFAULT_MODEL})")
    model_group.add_argument("--model-name", type=str, default=None,
                            help="自定义模型名称 (覆盖预设)")
    model_group.add_argument("--max-length", type=int, default=None,
                            help="最大序列长度 (默认: 根据预设)")
    
    # 数据参数
    data_group = parser.add_argument_group("数据配置")
    data_group.add_argument("--data-dir", type=str, default="data/processed",
                           help="处理后的数据目录")
    data_group.add_argument("--attack-type", type=str, default="sqli",
                           choices=["sqli", "xss", "rce"],
                           help="攻击类型 (默认: sqli)")
    
    # 训练参数
    train_group = parser.add_argument_group("训练配置")
    train_group.add_argument("--output-dir", type=str, default=None,
                            help="输出目录 (默认: 自动生成)")
    train_group.add_argument("--epochs", type=int, default=3,
                            help="训练轮数 (默认: 3)")
    train_group.add_argument("--batch-size", type=int, default=None,
                            help="batch size (默认: 根据预设)")
    train_group.add_argument("--gradient-accumulation", type=int, default=None,
                            help="梯度累积步数 (默认: 根据预设)")
    train_group.add_argument("--learning-rate", type=float, default=2e-5,
                            help="学习率 (默认: 2e-5)")
    
    # 优化参数
    optim_group = parser.add_argument_group("优化配置")
    optim_group.add_argument("--bf16", action="store_true", default=True,
                            help="使用 BF16 精度 (推荐，默认开启)")
    optim_group.add_argument("--no-bf16", action="store_true",
                            help="禁用 BF16，使用 FP16")
    optim_group.add_argument("--load-in-4bit", action="store_true",
                            help="使用 4-bit 量化")
    optim_group.add_argument("--flash-attention", action="store_true", default=True,
                            help="使用 Flash Attention 2")
    optim_group.add_argument("--gradient-checkpointing", action="store_true",
                            help="启用梯度检查点 (节省显存)")
    optim_group.add_argument("--optim", type=str, default="adamw_torch",
                            help="优化器类型 (默认: adamw_torch)")
    
    # 日志参数
    log_group = parser.add_argument_group("日志配置")
    log_group.add_argument("--logging-steps", type=int, default=50,
                          help="日志记录步数")
    log_group.add_argument("--save-steps", type=int, default=500,
                          help="模型保存步数")
    log_group.add_argument("--eval-steps", type=int, default=500,
                          help="评估步数")
    
    # 断点续训
    resume_group = parser.add_argument_group("断点续训")
    resume_group.add_argument("--resume", action="store_true", default=True,
                             help="自动从最新 checkpoint 恢复训练 (默认开启)")
    resume_group.add_argument("--no-resume", action="store_true",
                             help="禁用自动恢复，从头开始训练")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 处理 resume 参数
    if args.no_resume:
        args.resume = False
    
    # 处理 bf16 参数
    if args.no_bf16:
        args.bf16 = False
    
    # 设置默认输出目录
    if args.output_dir is None:
        preset_name = args.model_preset.replace(".", "_").replace("-", "_")
        args.output_dir = f"models/pretrain_{args.attack_type}_{preset_name}"
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    train(args)


if __name__ == "__main__":
    main()
