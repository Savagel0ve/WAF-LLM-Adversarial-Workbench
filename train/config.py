"""
训练配置文件 - 支持多种 GPU 配置
- 本地: RTX 4070 8GB
- 服务器: RTX 4090D 24GB
支持 Qwen2.5-Coder 和其他现代 LLM
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


# ==================== 模型预设 ====================
MODEL_PRESETS = {
    # Qwen2.5-Coder 系列 (推荐)
    "qwen2.5-coder-0.5b": {
        "model_name": "Qwen/Qwen2.5-Coder-0.5B",
        "max_length": 128,  # 减小序列长度
        "batch_size": 16,   # 安全的 batch size
        "gradient_accumulation": 2,  # 等效 batch=32
        "use_flash_attention": False,
    },
    "qwen2.5-coder-1.5b": {
        "model_name": "Qwen/Qwen2.5-Coder-1.5B",
        "max_length": 256,
        "batch_size": 4,    # 默认小 batch (8GB 显存)
        "gradient_accumulation": 8,
        "use_flash_attention": False,
        "gradient_checkpointing": True,
    },
    # RTX 4090 24GB 优化配置
    "qwen2.5-coder-1.5b-server": {
        "model_name": "Qwen/Qwen2.5-Coder-1.5B",
        "max_length": 512,   # 增加序列长度 (24GB 充足)
        "batch_size": 24,    # RTX 4090 大 batch (24GB 显存)
        "gradient_accumulation": 4,  # 等效 batch=96
        "use_flash_attention": True,  # RTX 4090 完美支持
        "gradient_checkpointing": False,
    },
    "qwen2.5-coder-3b-server": {
        "model_name": "Qwen/Qwen2.5-Coder-3B",
        "max_length": 256,   # 减小序列长度节省显存
        "batch_size": 8,     # RTX 4090 3B 模型适配 24GB
        "gradient_accumulation": 12, # 等效 batch=96
        "use_flash_attention": True,  # RTX 4090 完美支持
        "gradient_checkpointing": True,  # 启用梯度检查点节省显存
    },
    # RTX 4090 专属 - 7B 模型也可以跑
    "qwen2.5-coder-7b-server": {
        "model_name": "Qwen/Qwen2.5-Coder-7B",
        "max_length": 256,
        "batch_size": 4,
        "gradient_accumulation": 16,  # 等效 batch=64
        "use_flash_attention": True,
        "gradient_checkpointing": True,  # 7B 需要梯度检查点
    },
    "qwen2.5-coder-3b": {
        "model_name": "Qwen/Qwen2.5-Coder-3B",
        "max_length": 256,
        "batch_size": 2,
        "gradient_accumulation": 16,
        "use_flash_attention": False,
        "load_in_4bit": True,  # Requires quantization
    },
    # Qwen2.5 general series
    "qwen2.5-0.5b": {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "max_length": 512,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "use_flash_attention": False,
    },
    "qwen2.5-1.5b": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "max_length": 256,
        "batch_size": 4,
        "gradient_accumulation": 8,
        "use_flash_attention": False,
    },
    # DeepSeek Coder
    "deepseek-coder-1.3b": {
        "model_name": "deepseek-ai/deepseek-coder-1.3b-base",
        "max_length": 256,
        "batch_size": 4,
        "gradient_accumulation": 8,
        "use_flash_attention": False,
    },
    # Phi-3
    "phi-3-mini": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "max_length": 256,
        "batch_size": 2,
        "gradient_accumulation": 16,
        "use_flash_attention": False,
        "load_in_4bit": True,
    },
    # 旧版 GPT-2 (兼容)
    "gpt2": {
        "model_name": "gpt2",
        "max_length": 128,
        "batch_size": 4,
        "gradient_accumulation": 8,
        "use_flash_attention": False,
    },
    "gpt2-medium": {
        "model_name": "gpt2-medium",
        "max_length": 128,
        "batch_size": 2,
        "gradient_accumulation": 16,
        "use_flash_attention": False,
    },
}

# 默认模型
DEFAULT_MODEL = "qwen2.5-coder-1.5b"


@dataclass
class GPUConfig:
    """GPU和显存配置"""
    # 显存限制
    total_memory_gb: float = 8.0
    threshold_gb: float = 7.5  # 警告阈值
    
    # 显存优化
    use_fp16: bool = True  # 混合精度训练(节省50%显存)
    use_bf16: bool = False  # BF16 (Ampere+ GPU，更稳定)
    use_8bit_optimizer: bool = True  # 8-bit优化器(节省30-40%显存)
    gradient_checkpointing: bool = False  # 梯度检查点(节省显存但降低速度)
    
    # 量化配置
    load_in_4bit: bool = False  # 4-bit 量化 (需要 bitsandbytes)
    load_in_8bit: bool = False  # 8-bit 量化
    bnb_4bit_compute_dtype: str = "float16"  # 4-bit 计算精度
    bnb_4bit_quant_type: str = "nf4"  # 量化类型
    
    # Flash Attention
    use_flash_attention: bool = True  # 使用 Flash Attention 2 (需要支持)
    
    # DeepSpeed配置(可选)
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型选择 - 默认使用 Qwen2.5-Coder-1.5B
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    model_preset: str = DEFAULT_MODEL  # 使用预设配置
    
    # 序列长度
    max_length: int = 256
    
    # Tokenizer
    tokenizer_name: Optional[str] = None  # 默认与model_name相同
    trust_remote_code: bool = True  # Qwen 模型需要
    
    # 特殊token配置
    pad_token: Optional[str] = None  # 自动处理
    
    # 攻击类型
    attack_types: List[str] = field(default_factory=lambda: ["sqli", "xss", "rce"])
    
    @classmethod
    def from_preset(cls, preset_name: str):
        """从预设创建配置"""
        if preset_name not in MODEL_PRESETS:
            raise ValueError(f"未知的模型预设: {preset_name}, 可用: {list(MODEL_PRESETS.keys())}")
        
        preset = MODEL_PRESETS[preset_name]
        return cls(
            model_name=preset["model_name"],
            model_preset=preset_name,
            max_length=preset.get("max_length", 256),
        )


@dataclass
class TrainingConfig:
    """预训练配置 - 针对8GB显存优化，支持 Qwen2.5-Coder"""
    # 输出目录
    output_dir: str = "models/pretrain_sqli_qwen"
    
    # 训练超参数 - Qwen2.5-Coder-1.5B 优化
    num_train_epochs: int = 3  # Qwen 收敛更快
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # 等效batch_size = 32
    
    # 优化器
    learning_rate: float = 2e-5  # Qwen 推荐较小学习率
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # Qwen 推荐
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1  # 使用比例而非固定步数
    warmup_steps: int = 0  # 如果 > 0 则覆盖 warmup_ratio
    
    # 日志和保存
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 3
    
    # 评估
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # 数据加载 - 根据系统自动调整
    # Windows: 2, Linux 服务器: 8
    dataloader_num_workers: int = 8 if os.name != 'nt' else 2
    dataloader_pin_memory: bool = True
    
    # 混合精度 - 优先使用 bf16
    fp16: bool = False  # Qwen 推荐使用 bf16
    bf16: bool = True   # Ampere+ GPU 支持
    fp16_opt_level: str = "O2"
    
    # 优化器选择
    optim: str = "adamw_torch"  # 或 "adamw_bnb_8bit" 节省显存
    
    # 随机种子
    seed: int = 42
    
    @classmethod
    def from_model_preset(cls, preset_name: str, attack_type: str = "sqli"):
        """根据模型预设创建训练配置"""
        if preset_name not in MODEL_PRESETS:
            preset_name = DEFAULT_MODEL
        
        preset = MODEL_PRESETS[preset_name]
        
        return cls(
            output_dir=f"models/pretrain_{attack_type}_{preset_name.replace('.', '_').replace('-', '_')}",
            per_device_train_batch_size=preset.get("batch_size", 4),
            gradient_accumulation_steps=preset.get("gradient_accumulation", 8),
        )


@dataclass
class PPOConfig:
    """PPO强化学习配置 - 针对8GB显存优化，支持 Qwen2.5-Coder"""
    # 模型路径
    model_name: str = "models/pretrain_sqli_qwen"
    ref_model_name: Optional[str] = None
    
    # Batch配置 - Qwen 1.5B 优化
    batch_size: int = 128  # 每轮生成的样本数
    mini_batch_size: int = 8  # PPO更新的mini batch
    gradient_accumulation_steps: int = 4
    
    # PPO超参数 (论文推荐)
    learning_rate: float = 1.4e-5
    ppo_epochs: int = 4
    
    # KL散度约束 (关键!)
    init_kl_coef: float = 0.2  # Beta参数
    target_kl: float = 0.1
    adap_kl_ctrl: bool = False  # 论文使用固定beta
    
    # 奖励配置
    gamma: float = 0.99
    lam: float = 0.95
    
    # 优化
    optimize_cuda_cache: bool = True
    max_grad_norm: float = 1.0
    
    # 生成配置
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # 日志
    log_with: Optional[str] = None  # "tensorboard" 或 "wandb"
    seed: int = 42
    
    # 更新策略
    update_batch_size: int = 2  # 策略更新小批量


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "gptfuzzer-main/Datasets"
    grammar_dir: str = "gptfuzzer-main/grammar"
    
    # 数据集分割
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # 处理后的数据保存路径
    processed_dir: str = "data/processed"
    
    # 攻击类型
    attack_types: List[str] = field(default_factory=lambda: ["sqli", "xss", "rce"])


@dataclass
class WAFConfig:
    """WAF环境配置"""
    # WAF类型
    waf_type: str = "modsecurity"  # modsecurity 或 naxsi
    
    # WAF URL
    modsecurity_url: str = "http://localhost:8001"
    naxsi_url: str = "http://localhost:8002"
    
    # 请求配置
    timeout: int = 10
    max_retries: int = 3
    
    # 请求预算
    request_budget: int = 5000  # 每次训练的最大WAF请求次数


# 创建默认配置实例
def get_default_configs(model_preset: str = DEFAULT_MODEL):
    """获取默认配置"""
    model_config = ModelConfig.from_preset(model_preset)
    preset = MODEL_PRESETS.get(model_preset, MODEL_PRESETS[DEFAULT_MODEL])
    
    gpu_config = GPUConfig(
        load_in_4bit=preset.get("load_in_4bit", False),
        use_flash_attention=preset.get("use_flash_attention", True),
    )
    
    return {
        "gpu": gpu_config,
        "model": model_config,
        "training": TrainingConfig.from_model_preset(model_preset),
        "ppo": PPOConfig(),
        "data": DataConfig(),
        "waf": WAFConfig()
    }


def get_quantization_config(gpu_config: GPUConfig):
    """获取量化配置 (用于 bitsandbytes)"""
    if not gpu_config.load_in_4bit and not gpu_config.load_in_8bit:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        import torch
        
        if gpu_config.load_in_4bit:
            compute_dtype = getattr(torch, gpu_config.bnb_4bit_compute_dtype)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=gpu_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif gpu_config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        print("警告: bitsandbytes 未安装，无法使用量化")
        return None
    
    return None


if __name__ == "__main__":
    # 打印配置示例
    print("="*60)
    print("可用模型预设:")
    print("="*60)
    for name, preset in MODEL_PRESETS.items():
        quant = " (4-bit)" if preset.get("load_in_4bit") else ""
        print(f"  - {name}: {preset['model_name']}{quant}")
    
    print(f"\n默认模型: {DEFAULT_MODEL}")
    
    configs = get_default_configs()
    
    print("\n" + "="*60)
    print(f"默认训练配置 (RTX 4070 8GB) - {DEFAULT_MODEL}")
    print("="*60)
    
    print("\n📊 GPU配置:")
    print(f"  - 显存限制: {configs['gpu'].total_memory_gb} GB")
    print(f"  - FP16: {configs['gpu'].use_fp16}")
    print(f"  - BF16: {configs['gpu'].use_bf16}")
    print(f"  - 4-bit量化: {configs['gpu'].load_in_4bit}")
    print(f"  - Flash Attention: {configs['gpu'].use_flash_attention}")
    
    print("\n🤖 模型配置:")
    print(f"  - 模型: {configs['model'].model_name}")
    print(f"  - 预设: {configs['model'].model_preset}")
    print(f"  - 最大长度: {configs['model'].max_length}")
    
    print("\n🎓 训练配置:")
    print(f"  - Batch size: {configs['training'].per_device_train_batch_size}")
    print(f"  - Accumulation: {configs['training'].gradient_accumulation_steps}")
    print(f"  - 等效batch size: {configs['training'].per_device_train_batch_size * configs['training'].gradient_accumulation_steps}")
    print(f"  - 学习率: {configs['training'].learning_rate}")
    print(f"  - Epochs: {configs['training'].num_train_epochs}")
    print(f"  - BF16: {configs['training'].bf16}")
    
    print("\n🔄 PPO配置:")
    print(f"  - Batch size: {configs['ppo'].batch_size}")
    print(f"  - Mini batch: {configs['ppo'].mini_batch_size}")
    print(f"  - KL系数: {configs['ppo'].init_kl_coef}")
    print(f"  - 学习率: {configs['ppo'].learning_rate}")
