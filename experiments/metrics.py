"""
评估指标计算
============

实现论文中定义的评估指标:
- TP (True Positives): 绕过payload数量
- ER (Effective Rate): 有效率
- NRR (Non-Repetition Rate): 不重复率
- TER (Total Effective Rate): 总有效率
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json
from datetime import datetime


@dataclass
class ExperimentMetrics:
    """实验指标数据类"""
    
    # 基础统计
    total_generated: int = 0          # 总生成数
    unique_payloads: int = 0          # 不重复数
    valid_payloads: int = 0           # 语法有效数
    tested_payloads: int = 0          # 测试数
    
    # 绕过统计
    bypassed: int = 0                 # 绕过数
    blocked: int = 0                  # 阻止数
    errors: int = 0                   # 错误数
    
    # 奖励模型阶段统计 (用于计算TER)
    reward_phase_samples: int = 0     # 奖励模型训练样本数
    reward_phase_bypassed: int = 0    # 奖励模型阶段绕过数
    
    # 功能性验证
    functional_tested: int = 0        # 功能验证测试数
    functional_success: int = 0       # 功能验证成功数
    
    # 计算指标
    tp: int = 0                       # True Positives
    er: float = 0.0                   # Effective Rate
    nrr: float = 0.0                  # Non-Repetition Rate
    ter: float = 0.0                  # Total Effective Rate
    bypass_rate: float = 0.0          # 绕过率
    valid_rate: float = 0.0           # 有效率
    functional_rate: float = 0.0      # 功能性成功率
    
    # 时间统计
    generation_time: float = 0.0      # 生成时间(秒)
    testing_time: float = 0.0         # 测试时间(秒)
    tsr: float = 0.0                  # Time Spent per Request
    
    # 元数据
    timestamp: str = ""
    attack_type: str = ""
    waf_type: str = ""
    method: str = ""
    
    def __post_init__(self):
        """计算衍生指标"""
        self.calculate_metrics()
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def calculate_metrics(self):
        """计算所有指标"""
        # TP
        self.tp = self.bypassed
        
        # ER = TP_gen / N_distinct
        if self.unique_payloads > 0:
            self.er = self.bypassed / self.unique_payloads
        
        # NRR = N_distinct / N_total
        if self.total_generated > 0:
            self.nrr = self.unique_payloads / self.total_generated
        
        # TER = (TP_reward + TP_gen) / (|Dr| + N_distinct)
        total_samples = self.reward_phase_samples + self.unique_payloads
        total_bypassed = self.reward_phase_bypassed + self.bypassed
        if total_samples > 0:
            self.ter = total_bypassed / total_samples
        
        # 绕过率
        if self.tested_payloads > 0:
            self.bypass_rate = self.bypassed / self.tested_payloads
        
        # 有效率
        if self.unique_payloads > 0:
            self.valid_rate = self.valid_payloads / self.unique_payloads
        
        # 功能性成功率
        if self.functional_tested > 0:
            self.functional_rate = self.functional_success / self.functional_tested
        
        # TSR
        total_time = self.generation_time + self.testing_time
        if self.tested_payloads > 0:
            self.tsr = total_time / self.tested_payloads
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "basic_stats": {
                "total_generated": self.total_generated,
                "unique_payloads": self.unique_payloads,
                "valid_payloads": self.valid_payloads,
                "tested_payloads": self.tested_payloads,
            },
            "bypass_stats": {
                "bypassed": self.bypassed,
                "blocked": self.blocked,
                "errors": self.errors,
            },
            "reward_phase": {
                "samples": self.reward_phase_samples,
                "bypassed": self.reward_phase_bypassed,
            },
            "functional": {
                "tested": self.functional_tested,
                "success": self.functional_success,
            },
            "metrics": {
                "tp": self.tp,
                "er": self.er,
                "nrr": self.nrr,
                "ter": self.ter,
                "bypass_rate": self.bypass_rate,
                "valid_rate": self.valid_rate,
                "functional_rate": self.functional_rate,
            },
            "timing": {
                "generation_time": self.generation_time,
                "testing_time": self.testing_time,
                "tsr": self.tsr,
            },
            "metadata": {
                "timestamp": self.timestamp,
                "attack_type": self.attack_type,
                "waf_type": self.waf_type,
                "method": self.method,
            }
        }
    
    def save(self, path: str):
        """保存到JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentMetrics':
        """从字典创建"""
        flat_data = {}
        for section in data.values():
            if isinstance(section, dict):
                flat_data.update(section)
        return cls(**flat_data)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentMetrics':
        """从JSON文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ==================== 指标计算函数 ====================

def calculate_tp(bypassed_payloads: List[str]) -> int:
    """
    计算True Positives (绕过payload数量)
    
    Args:
        bypassed_payloads: 绕过的payload列表
        
    Returns:
        TP值
    """
    return len(bypassed_payloads)


def calculate_er(bypassed_count: int, unique_count: int) -> float:
    """
    计算Effective Rate (有效率)
    
    ER = TP_gen / N_distinct
    
    Args:
        bypassed_count: 绕过数量
        unique_count: 不重复payload数量
        
    Returns:
        ER值
    """
    if unique_count == 0:
        return 0.0
    return bypassed_count / unique_count


def calculate_nrr(unique_count: int, total_count: int) -> float:
    """
    计算Non-Repetition Rate (不重复率)
    
    NRR = N_distinct / N_total
    
    Args:
        unique_count: 不重复payload数量
        total_count: 总生成数量
        
    Returns:
        NRR值
    """
    if total_count == 0:
        return 0.0
    return unique_count / total_count


def calculate_ter(
    reward_bypassed: int,
    gen_bypassed: int,
    reward_samples: int,
    unique_payloads: int
) -> float:
    """
    计算Total Effective Rate (总有效率)
    
    TER = (TP_reward + TP_gen) / (|Dr| + N_distinct)
    
    Args:
        reward_bypassed: 奖励模型阶段绕过数
        gen_bypassed: 生成阶段绕过数
        reward_samples: 奖励模型训练样本数
        unique_payloads: 不重复payload数
        
    Returns:
        TER值
    """
    total_samples = reward_samples + unique_payloads
    if total_samples == 0:
        return 0.0
    return (reward_bypassed + gen_bypassed) / total_samples


def calculate_perplexity(log_probs: List[float]) -> float:
    """
    计算困惑度 (Perplexity)
    
    PPL = exp(-1/N * sum(log_probs))
    
    Args:
        log_probs: 对数概率列表
        
    Returns:
        困惑度
    """
    if not log_probs:
        return float('inf')
    avg_log_prob = sum(log_probs) / len(log_probs)
    return np.exp(-avg_log_prob)


def calculate_diversity(payloads: List[str]) -> Dict[str, float]:
    """
    计算payload多样性指标
    
    Args:
        payloads: payload列表
        
    Returns:
        多样性指标字典
    """
    if not payloads:
        return {"unique_ratio": 0.0, "avg_length": 0.0, "length_std": 0.0}
    
    # 唯一率
    unique_payloads = set(payloads)
    unique_ratio = len(unique_payloads) / len(payloads)
    
    # 长度统计
    lengths = [len(p) for p in payloads]
    avg_length = np.mean(lengths)
    length_std = np.std(lengths)
    
    # n-gram多样性
    def get_ngrams(text: str, n: int) -> List[str]:
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    all_bigrams = []
    all_trigrams = []
    for p in payloads:
        all_bigrams.extend(get_ngrams(p, 2))
        all_trigrams.extend(get_ngrams(p, 3))
    
    bigram_diversity = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    trigram_diversity = len(set(all_trigrams)) / max(len(all_trigrams), 1)
    
    return {
        "unique_ratio": unique_ratio,
        "avg_length": avg_length,
        "length_std": length_std,
        "bigram_diversity": bigram_diversity,
        "trigram_diversity": trigram_diversity,
    }


# ==================== 检查点记录 ====================

@dataclass
class CheckpointMetrics:
    """检查点指标记录"""
    
    request_count: int
    tp: int
    er: float
    nrr: float
    elapsed_time: float
    
    def to_dict(self) -> Dict:
        return {
            "requests": self.request_count,
            "tp": self.tp,
            "er": self.er,
            "nrr": self.nrr,
            "elapsed_time": self.elapsed_time,
        }


class MetricsTracker:
    """指标追踪器，用于记录实验过程中的指标变化"""
    
    def __init__(self, checkpoints: List[int]):
        """
        初始化追踪器
        
        Args:
            checkpoints: 检查点列表 (请求数)
        """
        self.checkpoints = sorted(checkpoints)
        self.checkpoint_metrics: List[CheckpointMetrics] = []
        self.current_request = 0
        self.bypassed_payloads: List[str] = []
        self.all_payloads: List[str] = []
        self.unique_payloads: set = set()
        self.start_time: Optional[float] = None
    
    def start(self):
        """开始追踪"""
        import time
        self.start_time = time.time()
    
    def add_result(self, payload: str, bypassed: bool):
        """
        添加单个结果
        
        Args:
            payload: 测试的payload
            bypassed: 是否绕过
        """
        self.current_request += 1
        self.all_payloads.append(payload)
        self.unique_payloads.add(payload)
        
        if bypassed:
            self.bypassed_payloads.append(payload)
        
        # 检查是否到达检查点
        if self.current_request in self.checkpoints:
            self._record_checkpoint()
    
    def add_batch_results(self, payloads: List[str], bypassed_flags: List[bool]):
        """
        添加批量结果
        
        Args:
            payloads: payload列表
            bypassed_flags: 对应的绕过标志列表
        """
        for payload, bypassed in zip(payloads, bypassed_flags):
            self.add_result(payload, bypassed)
    
    def _record_checkpoint(self):
        """记录当前检查点"""
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        tp = len(self.bypassed_payloads)
        unique_count = len(self.unique_payloads)
        total_count = len(self.all_payloads)
        
        checkpoint = CheckpointMetrics(
            request_count=self.current_request,
            tp=tp,
            er=calculate_er(tp, unique_count),
            nrr=calculate_nrr(unique_count, total_count),
            elapsed_time=elapsed,
        )
        
        self.checkpoint_metrics.append(checkpoint)
    
    def get_results(self) -> Dict:
        """获取追踪结果"""
        return {
            "checkpoints": [c.to_dict() for c in self.checkpoint_metrics],
            "final": {
                "total_requests": self.current_request,
                "tp": len(self.bypassed_payloads),
                "unique_payloads": len(self.unique_payloads),
                "er": calculate_er(len(self.bypassed_payloads), len(self.unique_payloads)),
                "nrr": calculate_nrr(len(self.unique_payloads), len(self.all_payloads)),
            }
        }
    
    def get_tp_curve(self) -> Tuple[List[int], List[int]]:
        """获取TP曲线数据"""
        requests = [c.request_count for c in self.checkpoint_metrics]
        tps = [c.tp for c in self.checkpoint_metrics]
        return requests, tps


# ==================== 结果比较 ====================

def compare_methods(
    results: Dict[str, ExperimentMetrics],
    baseline: str = "random_fuzzer"
) -> Dict[str, Dict[str, float]]:
    """
    比较不同方法的效果
    
    Args:
        results: 方法名 -> 指标 的字典
        baseline: 基线方法名
        
    Returns:
        相对改进比例
    """
    if baseline not in results:
        raise ValueError(f"基线方法 {baseline} 不在结果中")
    
    baseline_metrics = results[baseline]
    comparisons = {}
    
    for method, metrics in results.items():
        if method == baseline:
            continue
        
        comparisons[method] = {
            "tp_improvement": (
                metrics.tp / baseline_metrics.tp 
                if baseline_metrics.tp > 0 else float('inf')
            ),
            "er_improvement": (
                metrics.er / baseline_metrics.er 
                if baseline_metrics.er > 0 else float('inf')
            ),
            "nrr_improvement": (
                metrics.nrr / baseline_metrics.nrr 
                if baseline_metrics.nrr > 0 else float('inf')
            ),
        }
    
    return comparisons


if __name__ == "__main__":
    # 测试指标计算
    print("测试指标计算:")
    
    metrics = ExperimentMetrics(
        total_generated=10000,
        unique_payloads=8500,
        valid_payloads=8400,
        tested_payloads=8400,
        bypassed=420,
        blocked=7980,
        errors=0,
        reward_phase_samples=4000,
        reward_phase_bypassed=200,
        attack_type="sqli",
        waf_type="modsecurity",
        method="gptfuzzer",
    )
    
    print(f"TP: {metrics.tp}")
    print(f"ER: {metrics.er:.4f}")
    print(f"NRR: {metrics.nrr:.4f}")
    print(f"TER: {metrics.ter:.4f}")
    print(f"Bypass Rate: {metrics.bypass_rate:.4f}")
