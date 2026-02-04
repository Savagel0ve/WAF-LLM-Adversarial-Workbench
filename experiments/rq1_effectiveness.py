"""
RQ1: GPTFuzzer是否有效且高效？
==============================

实验目标:
对比GPTFuzzer与基线方法(Random Fuzzer, Grammar-based RL)的TP和效率。

评估指标:
- TP (True Positives): 绕过payload数量
- ER (Effective Rate): 有效率
- NRR (Non-Repetition Rate): 不重复率
- TSR (Time Spent per Request): 每请求耗时
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
from tqdm import tqdm

import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import (
    RQ1Config,
    get_model_path,
    get_data_path,
    CHECKPOINT_INTERVALS,
)
from experiments.experiment_runner import ExperimentRunner
from experiments.metrics import ExperimentMetrics, MetricsTracker
from experiments.baselines.random_fuzzer import RandomFuzzer
from experiments.baselines.grammar_fuzzer import create_grammar_fuzzer, create_grammar_rl


class RQ1Experiment:
    """RQ1实验: 有效性与效率对比"""
    
    def __init__(self, config: RQ1Config):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.runner = ExperimentRunner(config)
        self.results: Dict[str, Dict] = {}
        
        # 创建结果目录
        self.figures_dir = self.runner.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_gptfuzzer_model(self, attack_type: str):
        """加载GPTFuzzer模型"""
        model_path = get_model_path(attack_type, "rl", self.config.models_dir)
        if not os.path.exists(model_path):
            print(f"警告: GPTFuzzer模型不存在: {model_path}")
            return None, None
        return self.runner.load_model(model_path, "causal_lm")
    
    def _create_generator(
        self,
        method: str,
        attack_type: str
    ) -> Callable[[int], List[str]]:
        """
        创建生成器函数
        
        Args:
            method: 方法名称
            attack_type: 攻击类型
            
        Returns:
            生成函数 (batch_size) -> List[str]
        """
        if method == "gptfuzzer":
            # 加载GPTFuzzer模型
            model, tokenizer = self._load_gptfuzzer_model(attack_type)
            if model is None:
                # 回退到随机生成
                print(f"回退到随机生成 (模型不存在)")
                fuzzer = RandomFuzzer(attack_type)
                return fuzzer.generate
            
            def generate(batch_size: int) -> List[str]:
                return self.runner.generate_payloads(
                    model, tokenizer,
                    num_samples=batch_size,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature,
                )
            return generate
        
        elif method == "random_fuzzer":
            # 随机Fuzzer
            data_path = get_data_path(attack_type, "train", self.config.data_dir)
            fuzzer = RandomFuzzer(
                attack_type=attack_type,
                data_path=data_path,
                seed=self.config.seed,
            )
            return fuzzer.generate
        
        elif method == "grammar_rl":
            # 语法RL (需要先训练)
            grammar_rl = create_grammar_rl(attack_type, seed=self.config.seed)
            
            # 简单的奖励函数 (基于长度和特殊字符)
            def simple_reward(payload: str) -> float:
                score = 0.0
                if payload:
                    score += min(len(payload) / 100, 0.5)
                    special_chars = set("'\";<>|&`$()[]{}%")
                    special_count = sum(1 for c in payload if c in special_chars)
                    score += min(special_count / 10, 0.5)
                return score
            
            # 快速训练
            print(f"  训练Grammar-RL ({attack_type})...")
            grammar_rl.train(simple_reward, num_episodes=50, batch_size=32, verbose=False)
            
            return grammar_rl.generate
        
        else:
            raise ValueError(f"未知的方法: {method}")
    
    def run_single_experiment(
        self,
        attack_type: str,
        waf_type: str,
        method: str
    ) -> Dict:
        """
        运行单个实验
        
        Args:
            attack_type: 攻击类型
            waf_type: WAF类型
            method: 方法名称
            
        Returns:
            实验结果
        """
        print(f"\n{'='*60}")
        print(f"实验: {method} on {attack_type}/{waf_type}")
        print(f"{'='*60}")
        
        # 获取配置
        request_budget = self.config.request_budgets[attack_type]
        checkpoints = self.config.checkpoints[attack_type]
        waf_url = self.config.waf_urls[waf_type]
        
        # 创建生成器
        generator = self._create_generator(method, attack_type)
        
        # 初始化追踪器
        tracker = MetricsTracker(checkpoints)
        tracker.start()
        
        # 运行实验
        batch_size = self.config.batch_size
        total_tested = 0
        all_bypassed = []
        all_blocked = []
        
        start_time = time.time()
        
        pbar = tqdm(total=request_budget, desc=f"{method}")
        
        while total_tested < request_budget:
            # 生成
            current_batch = min(batch_size, request_budget - total_tested)
            payloads = generator(current_batch)
            
            # 测试WAF
            bypassed, blocked, errors = self.runner.test_waf(
                payloads, waf_url, waf_type
            )
            
            # 记录结果
            for payload in payloads:
                is_bypassed = payload in bypassed
                tracker.add_result(payload, is_bypassed)
                if is_bypassed:
                    all_bypassed.append(payload)
                else:
                    all_blocked.append(payload)
            
            total_tested += len(payloads)
            pbar.update(len(payloads))
        
        pbar.close()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 汇总结果
        tracker_results = tracker.get_results()
        
        result = {
            "experiment": "rq1_effectiveness",
            "attack_type": attack_type,
            "waf_type": waf_type,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "request_budget": request_budget,
                "batch_size": batch_size,
                "temperature": self.config.temperature,
                "max_length": self.config.max_length,
            },
            "checkpoints": tracker_results["checkpoints"],
            "final": {
                "total_requests": total_tested,
                "unique_payloads": tracker_results["final"]["unique_payloads"],
                "tp": tracker_results["final"]["tp"],
                "er": tracker_results["final"]["er"],
                "nrr": tracker_results["final"]["nrr"],
                "elapsed_time": elapsed_time,
                "tsr": elapsed_time / total_tested if total_tested > 0 else 0,
            },
            "bypassed_samples": all_bypassed[:20],  # 保存部分样例
        }
        
        # 打印结果
        print(f"\n结果:")
        print(f"  总请求: {total_tested}")
        print(f"  TP: {result['final']['tp']}")
        print(f"  ER: {result['final']['er']:.4f}")
        print(f"  NRR: {result['final']['nrr']:.4f}")
        print(f"  耗时: {elapsed_time:.2f}s")
        print(f"  TSR: {result['final']['tsr']*1000:.2f}ms")
        
        return result
    
    def run_all(self):
        """运行所有实验组合"""
        print("\n" + "="*80)
        print("RQ1: GPTFuzzer有效性与效率实验")
        print("="*80)
        
        for attack_type in self.config.attack_types:
            for waf_type in self.config.waf_types:
                for method in self.config.methods:
                    key = f"{attack_type}_{waf_type}_{method}"
                    
                    try:
                        result = self.run_single_experiment(
                            attack_type, waf_type, method
                        )
                        self.results[key] = result
                        
                        # 保存单个结果
                        filename = f"{attack_type}_{waf_type}_{method}.json"
                        self.runner.save_results(result, filename)
                        
                    except Exception as e:
                        print(f"实验失败 [{key}]: {e}")
                        import traceback
                        traceback.print_exc()
        
        # 保存汇总结果
        self._save_summary()
    
    def _save_summary(self):
        """保存汇总结果"""
        summary = {
            "experiment": "rq1_effectiveness",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "attack_types": self.config.attack_types,
                "waf_types": self.config.waf_types,
                "methods": self.config.methods,
                "request_budgets": self.config.request_budgets,
            },
            "results": {}
        }
        
        # 按攻击类型和WAF分组
        for attack_type in self.config.attack_types:
            for waf_type in self.config.waf_types:
                group_key = f"{attack_type}_{waf_type}"
                summary["results"][group_key] = {}
                
                for method in self.config.methods:
                    key = f"{attack_type}_{waf_type}_{method}"
                    if key in self.results:
                        summary["results"][group_key][method] = {
                            "tp": self.results[key]["final"]["tp"],
                            "er": self.results[key]["final"]["er"],
                            "nrr": self.results[key]["final"]["nrr"],
                            "tsr": self.results[key]["final"]["tsr"],
                        }
        
        self.runner.save_results(summary, "summary.json")
        
        print("\n" + "="*80)
        print("实验完成! 结果已保存到:", self.runner.results_dir)
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RQ1: 有效性与效率实验")
    
    parser.add_argument("--output_dir", type=str, default="results",
                        help="输出目录")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据目录")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="模型目录")
    parser.add_argument("--attack_types", type=str, nargs="+",
                        default=["sqli", "xss", "rce"],
                        help="攻击类型")
    parser.add_argument("--waf_types", type=str, nargs="+",
                        default=["modsecurity"],
                        help="WAF类型")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["gptfuzzer", "random_fuzzer", "grammar_rl"],
                        help="对比方法")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="批次大小")
    
    args = parser.parse_args()
    
    # 创建配置
    config = RQ1Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        attack_types=args.attack_types,
        waf_types=args.waf_types,
        methods=args.methods,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    
    # 运行实验
    experiment = RQ1Experiment(config)
    experiment.run_all()


if __name__ == "__main__":
    main()
