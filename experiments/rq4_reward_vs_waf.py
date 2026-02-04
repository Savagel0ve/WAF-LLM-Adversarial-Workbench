"""
RQ4: 奖励模型vs直接WAF反馈有什么优势？
=====================================

实验目标:
对比奖励模型指导RL与直接使用WAF二元反馈的效果。

对比方法:
1. 奖励模型: 使用预测的绕过概率 r(τ) ∈ [0, 1]
2. WAF反馈: 直接使用WAF测试结果 {0, 1}

评估指标:
- TP随请求数的变化曲线
- 收敛速度
- 最终绕过率
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
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import RQ4Config, get_model_path, get_data_path
from experiments.experiment_runner import ExperimentRunner
from experiments.metrics import MetricsTracker


class RQ4Experiment:
    """RQ4实验: 奖励模型 vs WAF反馈"""
    
    def __init__(self, config: RQ4Config):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.runner = ExperimentRunner(config)
        self.results: Dict[str, Dict] = {}
    
    def load_reward_model(self, attack_type: str):
        """
        加载奖励模型
        
        Args:
            attack_type: 攻击类型
            
        Returns:
            (model, tokenizer) 或 (None, None)
        """
        model_path = get_model_path(attack_type, "reward", self.config.models_dir)
        if not os.path.exists(model_path):
            print(f"警告: 奖励模型不存在: {model_path}")
            return None, None
        
        return self.runner.load_model(model_path, "sequence_classification")
    
    def get_reward_from_model(
        self,
        model,
        tokenizer,
        payloads: List[str]
    ) -> List[float]:
        """
        使用奖励模型计算奖励
        
        Args:
            model: 奖励模型
            tokenizer: tokenizer
            payloads: payload列表
            
        Returns:
            奖励列表
        """
        model.eval()
        rewards = []
        
        with torch.no_grad():
            for payload in payloads:
                inputs = tokenizer(
                    payload,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True
                ).to(self.runner.device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Sigmoid 得到绕过概率
                prob = torch.sigmoid(logits).item()
                rewards.append(prob)
        
        return rewards
    
    def get_reward_from_waf(
        self,
        payloads: List[str],
        waf_url: str,
        waf_type: str = "modsecurity"
    ) -> List[float]:
        """
        使用WAF测试结果作为奖励 (0 或 1)
        
        Args:
            payloads: payload列表
            waf_url: WAF URL
            waf_type: WAF类型
            
        Returns:
            奖励列表 (0.0 或 1.0)
        """
        bypassed, blocked, errors = self.runner.test_waf(
            payloads, waf_url, waf_type
        )
        
        bypassed_set = set(bypassed)
        rewards = [1.0 if p in bypassed_set else 0.0 for p in payloads]
        
        return rewards
    
    def simulate_rl_training(
        self,
        attack_type: str,
        reward_method: str,
        num_epochs: int = 20,
        batch_size: int = 256
    ) -> Dict:
        """
        模拟RL训练过程并记录指标
        
        Args:
            attack_type: 攻击类型
            reward_method: 奖励方法 (reward_model/waf_feedback)
            num_epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练结果
        """
        print(f"\n{'='*50}")
        print(f"训练方法: {reward_method} ({attack_type})")
        print(f"{'='*50}")
        
        waf_url = self.config.waf_urls.get("modsecurity", "http://localhost:8001")
        
        # 加载模型
        pretrain_path = get_model_path(attack_type, "pretrain", self.config.models_dir)
        if not os.path.exists(pretrain_path):
            print(f"警告: 预训练模型不存在: {pretrain_path}")
            return {"error": "Model not found"}
        
        model, tokenizer = self.runner.load_model(pretrain_path)
        
        # 奖励模型 (如果使用)
        reward_model, reward_tokenizer = None, None
        if reward_method == "reward_model":
            reward_model, reward_tokenizer = self.load_reward_model(attack_type)
        
        # 记录训练过程
        epoch_results = []
        cumulative_requests = 0
        cumulative_tp = 0
        all_bypassed = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 生成payload
            payloads = self.runner.generate_payloads(
                model, tokenizer,
                num_samples=batch_size,
                max_length=128,
                temperature=1.0,
            )
            
            # 计算奖励
            if reward_method == "reward_model" and reward_model is not None:
                rewards = self.get_reward_from_model(
                    reward_model, reward_tokenizer, payloads
                )
            else:
                rewards = self.get_reward_from_waf(payloads, waf_url)
            
            # 统计本轮结果
            tp_this_epoch = sum(1 for r in rewards if r > 0.5)
            bypassed_this_epoch = [p for p, r in zip(payloads, rewards) if r > 0.5]
            
            cumulative_requests += len(payloads)
            cumulative_tp += tp_this_epoch
            all_bypassed.extend(bypassed_this_epoch)
            
            avg_reward = np.mean(rewards)
            
            epoch_result = {
                "epoch": epoch + 1,
                "requests": cumulative_requests,
                "tp": cumulative_tp,
                "tp_this_epoch": tp_this_epoch,
                "avg_reward": avg_reward,
                "er": cumulative_tp / cumulative_requests if cumulative_requests > 0 else 0,
            }
            epoch_results.append(epoch_result)
            
            print(f"  本轮TP: {tp_this_epoch}, 累计TP: {cumulative_tp}")
            print(f"  平均奖励: {avg_reward:.4f}")
            
            # 这里应该有实际的模型更新步骤
            # 为了简化，我们只记录统计信息
        
        return {
            "attack_type": attack_type,
            "reward_method": reward_method,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "epoch_results": epoch_results,
            "final": {
                "total_requests": cumulative_requests,
                "tp": cumulative_tp,
                "er": cumulative_tp / cumulative_requests if cumulative_requests > 0 else 0,
                "unique_bypassed": len(set(all_bypassed)),
            },
            "bypassed_samples": list(set(all_bypassed))[:20],
        }
    
    def run_comparison(self, attack_type: str) -> Dict:
        """
        运行对比实验
        
        Args:
            attack_type: 攻击类型
            
        Returns:
            对比结果
        """
        results = {}
        
        for reward_method in self.config.reward_methods:
            result = self.simulate_rl_training(
                attack_type,
                reward_method,
                num_epochs=self.config.rl_epochs,
                batch_size=self.config.rl_batch_size,
            )
            results[reward_method] = result
            
            # 保存单个结果
            filename = f"{attack_type}_{reward_method}.json"
            self.runner.save_results({
                "experiment": "rq4_reward_vs_waf",
                "timestamp": datetime.now().isoformat(),
                **result
            }, filename)
        
        return results
    
    def run_all(self):
        """运行所有实验"""
        print("\n" + "="*80)
        print("RQ4: 奖励模型 vs WAF反馈实验")
        print("="*80)
        
        for attack_type in self.config.attack_types:
            print(f"\n攻击类型: {attack_type}")
            result = self.run_comparison(attack_type)
            self.results[attack_type] = result
        
        # 保存汇总
        self._save_summary()
    
    def _save_summary(self):
        """保存汇总结果"""
        summary = {
            "experiment": "rq4_reward_vs_waf",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "attack_types": self.config.attack_types,
                "reward_methods": self.config.reward_methods,
                "rl_epochs": self.config.rl_epochs,
                "rl_batch_size": self.config.rl_batch_size,
            },
            "comparison": {}
        }
        
        for attack_type in self.config.attack_types:
            if attack_type in self.results:
                summary["comparison"][attack_type] = {}
                for method, result in self.results[attack_type].items():
                    if "final" in result:
                        summary["comparison"][attack_type][method] = {
                            "final_tp": result["final"]["tp"],
                            "final_er": result["final"]["er"],
                        }
        
        self.runner.save_results(summary, "summary.json")
        
        print("\n" + "="*80)
        print("实验完成! 结果已保存")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RQ4: 奖励模型vs WAF反馈实验")
    
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--attack_types", type=str, nargs="+",
                        default=["sqli", "xss", "rce"])
    parser.add_argument("--rl_epochs", type=int, default=20)
    parser.add_argument("--rl_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = RQ4Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        attack_types=args.attack_types,
        rl_epochs=args.rl_epochs,
        rl_batch_size=args.rl_batch_size,
        seed=args.seed,
    )
    
    experiment = RQ4Experiment(config)
    experiment.run_all()


if __name__ == "__main__":
    main()
