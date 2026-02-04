"""
实验配置管理
============

管理6个RQ实验的配置参数，参考论文设置。
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import json


# ==================== 常量定义 ====================

ATTACK_TYPES = ["sqli", "xss", "rce"]
WAF_TYPES = ["modsecurity", "naxsi"]

# 论文中的请求预算
REQUEST_BUDGETS = {
    "sqli": 1_250_000,
    "xss": 35_000,
    "rce": 35_000,
}

# 预训练数据规模
PRETRAIN_DATA_SIZES = {
    "sqli": 512_000,
    "xss": 512_000,
    "rce": 37_302,
}

# 奖励模型训练数据规模
REWARD_DATA_SIZES = {
    "sqli": 4_000,
    "xss": 2_000,
    "rce": 2_000,
}

# 检查点间隔 (用于记录TP曲线)
CHECKPOINT_INTERVALS = {
    "sqli": [10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1250000],
    "xss": [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000],
    "rce": [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000],
}


# ==================== 基础配置 ====================

@dataclass
class ExperimentConfig:
    """实验基础配置"""
    
    # 实验标识
    experiment_name: str = "experiment"
    experiment_id: str = ""
    
    # 路径配置
    base_dir: str = "."
    output_dir: str = "results"
    models_dir: str = "models"
    data_dir: str = "data"
    
    # 攻击类型和WAF
    attack_types: List[str] = field(default_factory=lambda: ATTACK_TYPES.copy())
    waf_types: List[str] = field(default_factory=lambda: WAF_TYPES.copy())
    
    # WAF URL配置
    waf_urls: Dict[str, str] = field(default_factory=lambda: {
        "modsecurity": "http://localhost:8001",
        "naxsi": "http://localhost:8002",
    })
    
    # 随机种子
    seed: int = 42
    
    # 重复次数
    num_runs: int = 5
    
    # 设备
    device: str = "cuda"
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.experiment_id:
            import datetime
            self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        self.results_path = Path(self.output_dir) / self.experiment_name
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Optional[str] = None):
        """保存配置到JSON"""
        if path is None:
            path = self.results_path / "config.json"
        
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
        
        # 转换Path对象
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """从JSON加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ==================== RQ1: 有效性与效率对比 ====================

@dataclass
class RQ1Config(ExperimentConfig):
    """RQ1实验配置: GPTFuzzer有效性与效率"""
    
    experiment_name: str = "rq1_effectiveness"
    
    # 对比方法
    methods: List[str] = field(default_factory=lambda: [
        "gptfuzzer",
        "random_fuzzer", 
        "grammar_rl",
    ])
    
    # 请求预算
    request_budgets: Dict[str, int] = field(default_factory=lambda: REQUEST_BUDGETS.copy())
    
    # 检查点
    checkpoints: Dict[str, List[int]] = field(default_factory=lambda: CHECKPOINT_INTERVALS.copy())
    
    # GPTFuzzer配置
    pretrain_data_sizes: Dict[str, int] = field(default_factory=lambda: PRETRAIN_DATA_SIZES.copy())
    reward_data_sizes: Dict[str, int] = field(default_factory=lambda: REWARD_DATA_SIZES.copy())
    kl_coefficient: float = 0.2
    
    # 生成配置
    batch_size: int = 100
    max_length: int = 128
    temperature: float = 1.0


# ==================== RQ2: 攻击语法影响 ====================

@dataclass
class RQ2Config(ExperimentConfig):
    """RQ2实验配置: 攻击语法对效果的影响"""
    
    experiment_name: str = "rq2_grammar"
    
    # 序列类型
    sequence_types: List[str] = field(default_factory=lambda: ["short", "long"])
    
    # 评估配置
    num_payloads: int = 100_000  # 生成的payload数量
    
    # 效率测试
    measure_generation_time: bool = True


# ==================== RQ3: 预训练数据规模 ====================

@dataclass  
class RQ3Config(ExperimentConfig):
    """RQ3实验配置: 预训练数据规模影响"""
    
    experiment_name: str = "rq3_data_scale"
    
    # 数据规模设置
    data_scales: List[int] = field(default_factory=lambda: [0, 20_000, 256_000, 512_000])
    
    # 只测试SQLi和XSS (RCE数据集太小)
    attack_types: List[str] = field(default_factory=lambda: ["sqli", "xss"])
    
    # 测试请求数
    test_requests: Dict[str, int] = field(default_factory=lambda: {
        "sqli": 1_250_000,
        "xss": 35_000,
    })


# ==================== RQ4: 奖励模型 vs WAF反馈 ====================

@dataclass
class RQ4Config(ExperimentConfig):
    """RQ4实验配置: 奖励模型vs直接WAF反馈"""
    
    experiment_name: str = "rq4_reward_vs_waf"
    
    # 对比方法
    reward_methods: List[str] = field(default_factory=lambda: [
        "reward_model",  # 使用奖励模型
        "waf_feedback",  # 直接使用WAF反馈
    ])
    
    # RL训练配置
    rl_epochs: int = 20
    rl_batch_size: int = 256


# ==================== RQ5: 超参数影响 ====================

@dataclass
class RQ5Config(ExperimentConfig):
    """RQ5实验配置: 超参数影响"""
    
    experiment_name: str = "rq5_hyperparams"
    
    # KL散度系数测试值
    kl_coefficients: List[float] = field(default_factory=lambda: [0, 0.1, 0.2, 0.5, 1.0])
    
    # 奖励模型数据量测试值
    reward_data_sizes_test: List[int] = field(default_factory=lambda: [2000, 4000])
    
    # 评估指标
    metrics: List[str] = field(default_factory=lambda: ["er", "nrr", "ter"])


# ==================== RQ6: 超越语法能力 ====================

@dataclass
class RQ6Config(ExperimentConfig):
    """RQ6实验配置: 超越语法的能力"""
    
    experiment_name: str = "rq6_novel_payloads"
    
    # 生成payload数量
    num_generate: int = 500_000
    
    # 只测试SQLi (XSS/RCE空间太小)
    attack_types: List[str] = field(default_factory=lambda: ["sqli"])
    
    # 功能性验证样本数
    functional_verify_samples: int = 100


# ==================== 工具函数 ====================

def get_model_path(attack_type: str, stage: str, base_dir: str = "models") -> str:
    """
    获取模型路径
    
    Args:
        attack_type: 攻击类型 (sqli/xss/rce)
        stage: 训练阶段 (pretrain/reward/rl)
        base_dir: 模型基础目录
        
    Returns:
        模型路径
    """
    stage_dirs = {
        "pretrain": f"pretrain_{attack_type}_qwen2_5_coder_1_5b",
        "reward": f"reward_{attack_type}_qwen/final_reward_model",
        "rl": f"rl_{attack_type}_qwen",
    }
    return os.path.join(base_dir, stage_dirs.get(stage, stage))


def get_data_path(attack_type: str, data_type: str = "train", base_dir: str = "data") -> str:
    """
    获取数据路径
    
    Args:
        attack_type: 攻击类型 (sqli/xss/rce)
        data_type: 数据类型 (train/val/test)
        base_dir: 数据基础目录
        
    Returns:
        数据路径
    """
    return os.path.join(base_dir, "processed", attack_type, f"{data_type}.txt")


def create_experiment_config(rq: str, **kwargs) -> ExperimentConfig:
    """
    创建指定RQ的实验配置
    
    Args:
        rq: 研究问题编号 (rq1-rq6)
        **kwargs: 额外配置参数
        
    Returns:
        实验配置对象
    """
    config_classes = {
        "rq1": RQ1Config,
        "rq2": RQ2Config,
        "rq3": RQ3Config,
        "rq4": RQ4Config,
        "rq5": RQ5Config,
        "rq6": RQ6Config,
    }
    
    config_class = config_classes.get(rq.lower())
    if config_class is None:
        raise ValueError(f"未知的研究问题: {rq}, 可用: {list(config_classes.keys())}")
    
    return config_class(**kwargs)


if __name__ == "__main__":
    # 测试配置
    print("测试RQ1配置:")
    config = RQ1Config()
    print(f"  实验名称: {config.experiment_name}")
    print(f"  对比方法: {config.methods}")
    print(f"  请求预算: {config.request_budgets}")
    
    print("\n测试RQ5配置:")
    config = RQ5Config()
    print(f"  KL系数: {config.kl_coefficients}")
    print(f"  数据量: {config.reward_data_sizes_test}")
