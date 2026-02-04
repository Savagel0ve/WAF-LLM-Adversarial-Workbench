"""
GPTFuzzer 实验模块
==================

实验框架用于复现论文中的6个研究问题(RQ)实验。

模块结构:
- experiment_config.py: 实验配置管理
- experiment_runner.py: 统一实验运行器
- metrics.py: 评估指标计算
- analysis.py: 结果分析与可视化
- baselines/: 基线方法实现
- rq1~rq6: 各RQ实验脚本
"""

from .experiment_config import (
    ExperimentConfig,
    RQ1Config,
    RQ2Config,
    RQ3Config,
    RQ4Config,
    RQ5Config,
    RQ6Config,
    ATTACK_TYPES,
    WAF_TYPES,
)

from .metrics import (
    calculate_tp,
    calculate_er,
    calculate_nrr,
    calculate_ter,
    ExperimentMetrics,
)

__all__ = [
    "ExperimentConfig",
    "RQ1Config",
    "RQ2Config", 
    "RQ3Config",
    "RQ4Config",
    "RQ5Config",
    "RQ6Config",
    "ATTACK_TYPES",
    "WAF_TYPES",
    "calculate_tp",
    "calculate_er",
    "calculate_nrr",
    "calculate_ter",
    "ExperimentMetrics",
]
