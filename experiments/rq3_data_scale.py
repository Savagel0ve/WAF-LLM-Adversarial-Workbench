"""
RQ3: 不同预训练数据规模对GPTFuzzer有何影响？
============================================

实验目标:
探索预训练数据量对最终效果的影响。

数据规模设置:
- 0K: 无预训练 (直接RL)
- 20K: 小规模
- 256K: 中规模 (论文一半)
- 512K: 全规模 (论文配置)

评估指标:
- TP (True Positives): 绕过payload数量
- Valid Rate: 语法有效率
- Perplexity: 预训练困惑度
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import RQ3Config, get_data_path, get_model_path
from experiments.experiment_runner import ExperimentRunner
from experiments.metrics import ExperimentMetrics


class RQ3Experiment:
    """RQ3实验: 预训练数据规模影响"""
    
    def __init__(self, config: RQ3Config):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.runner = ExperimentRunner(config)
        self.results: Dict[str, Dict] = {}
    
    def prepare_scaled_data(
        self,
        attack_type: str,
        scale: int
    ) -> Optional[str]:
        """
        准备指定规模的数据集
        
        Args:
            attack_type: 攻击类型
            scale: 数据规模
            
        Returns:
            临时数据文件路径 (如果scale>0)
        """
        if scale == 0:
            return None
        
        # 读取原始数据
        data_path = get_data_path(attack_type, "train", self.config.data_dir)
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = [line.strip() for line in f if line.strip()]
        
        # 采样
        if scale >= len(all_data):
            sampled = all_data
        else:
            random.seed(self.config.seed)
            sampled = random.sample(all_data, scale)
        
        # 保存到临时文件
        temp_dir = self.runner.results_dir / "temp_data"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / f"{attack_type}_{scale}.txt"
        with open(temp_path, 'w', encoding='utf-8') as f:
            for line in sampled:
                f.write(line + '\n')
        
        print(f"  准备数据: {len(sampled)} 条 -> {temp_path}")
        return str(temp_path)
    
    def calculate_perplexity(
        self,
        model,
        tokenizer,
        test_data: List[str],
        max_samples: int = 1000
    ) -> float:
        """
        计算模型在测试集上的困惑度
        
        Args:
            model: 语言模型
            tokenizer: tokenizer
            test_data: 测试数据
            max_samples: 最大样本数
            
        Returns:
            困惑度
        """
        if max_samples < len(test_data):
            test_data = random.sample(test_data, max_samples)
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(test_data, desc="计算困惑度"):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.runner.device)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                
                num_tokens = inputs["input_ids"].numel()
                total_loss += loss * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_model(
        self,
        model,
        tokenizer,
        attack_type: str,
        waf_url: str,
        num_samples: int = 10000
    ) -> Dict:
        """
        评估模型
        
        Args:
            model: 模型
            tokenizer: tokenizer
            attack_type: 攻击类型
            waf_url: WAF URL
            num_samples: 评估样本数
            
        Returns:
            评估结果
        """
        # 生成payload
        payloads = self.runner.generate_payloads(
            model, tokenizer,
            num_samples=num_samples,
            max_length=128,
            temperature=1.0,
        )
        
        # 去重
        unique_payloads = list(set(payloads))
        
        # 测试WAF
        bypassed, blocked, errors = self.runner.test_waf(
            unique_payloads, waf_url
        )
        
        return {
            "total_generated": len(payloads),
            "unique_payloads": len(unique_payloads),
            "tp": len(bypassed),
            "blocked": len(blocked),
            "errors": len(errors),
            "bypass_rate": len(bypassed) / len(unique_payloads) if unique_payloads else 0,
            "nrr": len(unique_payloads) / len(payloads) if payloads else 0,
        }
    
    def run_scale_experiment(
        self,
        attack_type: str,
        scale: int
    ) -> Dict:
        """
        运行单个数据规模实验
        
        Args:
            attack_type: 攻击类型
            scale: 数据规模
            
        Returns:
            实验结果
        """
        print(f"\n{'='*50}")
        print(f"数据规模: {scale:,} ({attack_type})")
        print(f"{'='*50}")
        
        result = {
            "scale": scale,
            "attack_type": attack_type,
            "timestamp": datetime.now().isoformat(),
        }
        
        if scale == 0:
            # 无预训练，直接使用基础模型
            print("  无预训练模式 (直接RL)")
            result["perplexity"] = float('inf')
            result["note"] = "No pretraining, direct RL"
            result["evaluation"] = {
                "tp": 0,
                "bypass_rate": 0,
                "nrr": 0,
                "note": "Requires training from scratch"
            }
        else:
            # 准备数据
            data_path = self.prepare_scaled_data(attack_type, scale)
            
            # 检查是否有对应规模的预训练模型
            # 实际使用中需要先训练这些模型
            model_suffix = f"_{scale//1000}k" if scale >= 1000 else f"_{scale}"
            model_path = get_model_path(attack_type, "pretrain", self.config.models_dir)
            
            # 如果模型存在，进行评估
            if os.path.exists(model_path):
                print(f"  加载模型: {model_path}")
                model, tokenizer = self.runner.load_model(model_path)
                
                # 计算困惑度
                test_data = self.runner.load_data(attack_type, "test", max_samples=1000)
                perplexity = self.calculate_perplexity(model, tokenizer, test_data)
                result["perplexity"] = perplexity
                print(f"  困惑度: {perplexity:.2f}")
                
                # 评估WAF绕过
                waf_url = self.config.waf_urls.get("modsecurity", "http://localhost:8001")
                num_test = min(self.config.test_requests.get(attack_type, 10000), 10000)
                evaluation = self.evaluate_model(
                    model, tokenizer, attack_type, waf_url, num_test
                )
                result["evaluation"] = evaluation
                
                print(f"  TP: {evaluation['tp']}")
                print(f"  绕过率: {evaluation['bypass_rate']:.4f}")
            else:
                print(f"  警告: 模型不存在 {model_path}")
                result["perplexity"] = None
                result["evaluation"] = None
                result["note"] = f"Model not found: {model_path}"
        
        return result
    
    def run_all(self):
        """运行所有实验"""
        print("\n" + "="*80)
        print("RQ3: 预训练数据规模影响实验")
        print("="*80)
        print(f"数据规模: {self.config.data_scales}")
        
        for attack_type in self.config.attack_types:
            print(f"\n攻击类型: {attack_type}")
            self.results[attack_type] = {}
            
            for scale in self.config.data_scales:
                result = self.run_scale_experiment(attack_type, scale)
                self.results[attack_type][f"scale_{scale}"] = result
                
                # 保存单个结果
                filename = f"{attack_type}_scale_{scale}.json"
                self.runner.save_results(result, filename)
        
        # 保存汇总
        self._save_summary()
    
    def _save_summary(self):
        """保存汇总结果"""
        summary = {
            "experiment": "rq3_data_scale",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "data_scales": self.config.data_scales,
                "attack_types": self.config.attack_types,
            },
            "results": {}
        }
        
        # 按攻击类型汇总
        for attack_type in self.config.attack_types:
            summary["results"][attack_type] = []
            
            for scale in self.config.data_scales:
                key = f"scale_{scale}"
                if key in self.results.get(attack_type, {}):
                    result = self.results[attack_type][key]
                    summary["results"][attack_type].append({
                        "scale": scale,
                        "perplexity": result.get("perplexity"),
                        "tp": result.get("evaluation", {}).get("tp"),
                        "bypass_rate": result.get("evaluation", {}).get("bypass_rate"),
                    })
        
        self.runner.save_results(summary, "summary.json")
        
        print("\n" + "="*80)
        print("实验完成! 结果已保存")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RQ3: 预训练数据规模实验")
    
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--attack_types", type=str, nargs="+",
                        default=["sqli", "xss"])
    parser.add_argument("--data_scales", type=int, nargs="+",
                        default=[0, 20000, 256000, 512000])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = RQ3Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        attack_types=args.attack_types,
        data_scales=args.data_scales,
        seed=args.seed,
    )
    
    experiment = RQ3Experiment(config)
    experiment.run_all()


if __name__ == "__main__":
    main()
