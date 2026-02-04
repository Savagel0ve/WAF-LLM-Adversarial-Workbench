"""
RQ2: 攻击语法如何影响GPTFuzzer的效果和效率？
============================================

实验目标:
对比短序列(仅终结符)和长序列(含非终结符)的效果差异。

评估指标:
- TP (True Positives): 绕过payload数量
- TSR (Time Spent per Request): 每请求耗时
- 平均序列长度
- 生成速度 (tokens/sec)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import RQ2Config, get_model_path, get_data_path
from experiments.experiment_runner import ExperimentRunner
from experiments.metrics import ExperimentMetrics


class RQ2Experiment:
    """RQ2实验: 攻击语法影响"""
    
    def __init__(self, config: RQ2Config):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.runner = ExperimentRunner(config)
        self.results: Dict[str, Dict] = {}
    
    def prepare_data(self, attack_type: str, sequence_type: str) -> List[str]:
        """
        准备不同类型的序列数据
        
        Args:
            attack_type: 攻击类型
            sequence_type: 序列类型 (short/long)
            
        Returns:
            数据列表
        """
        # 短序列: 只包含终结符 (标准payload)
        # 长序列: 包含非终结符标记 (模拟语法树展开)
        
        data_path = get_data_path(attack_type, "train", self.config.data_dir)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            payloads = [line.strip() for line in f if line.strip()]
        
        if sequence_type == "long":
            # 为长序列添加语法标记
            processed = []
            for payload in payloads:
                # 简单模拟: 在关键位置插入非终结符标记
                long_payload = self._add_grammar_markers(payload, attack_type)
                processed.append(long_payload)
            return processed
        else:
            # 短序列直接返回
            return payloads
    
    def _add_grammar_markers(self, payload: str, attack_type: str) -> str:
        """
        为payload添加语法标记 (模拟长序列)
        
        Args:
            payload: 原始payload
            attack_type: 攻击类型
            
        Returns:
            带标记的payload
        """
        if attack_type == "sqli":
            # SQLi语法标记
            markers = {
                "SELECT": "<select_stmt> SELECT",
                "UNION": "<union_clause> UNION",
                "FROM": "<from_clause> FROM",
                "WHERE": "<where_clause> WHERE",
                "AND": "<operator> AND",
                "OR": "<operator> OR",
                "'": "<context> '",
            }
        elif attack_type == "xss":
            # XSS语法标记
            markers = {
                "<script": "<script_tag> <script",
                "<img": "<img_tag> <img",
                "onerror": "<event> onerror",
                "onclick": "<event> onclick",
                "alert": "<js_func> alert",
            }
        elif attack_type == "rce":
            # RCE语法标记
            markers = {
                ";": "<separator> ;",
                "|": "<pipe> |",
                "cat": "<command> cat",
                "ls": "<command> ls",
                "id": "<command> id",
            }
        else:
            markers = {}
        
        result = payload
        for key, value in markers.items():
            result = result.replace(key, value)
        
        return result
    
    def measure_generation_efficiency(
        self,
        model,
        tokenizer,
        num_samples: int = 1000,
        max_length: int = 128
    ) -> Dict:
        """
        测量生成效率
        
        Args:
            model: 生成模型
            tokenizer: tokenizer
            num_samples: 样本数
            max_length: 最大长度
            
        Returns:
            效率指标
        """
        start_time = time.time()
        total_tokens = 0
        lengths = []
        
        batch_size = 10
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        model.eval()
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="生成"):
                current_batch = min(batch_size, num_samples - len(lengths))
                
                # 生成
                input_ids = torch.tensor(
                    [[tokenizer.bos_token_id or tokenizer.eos_token_id]] * current_batch,
                    device=self.runner.device
                )
                
                outputs = model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
                for output in outputs:
                    length = len(output)
                    lengths.append(length)
                    total_tokens += length
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "total_samples": len(lengths),
            "total_tokens": total_tokens,
            "elapsed_time": elapsed,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
            "samples_per_second": len(lengths) / elapsed if elapsed > 0 else 0,
        }
    
    def run_comparison(self, attack_type: str) -> Dict:
        """
        运行序列类型对比实验
        
        Args:
            attack_type: 攻击类型
            
        Returns:
            对比结果
        """
        print(f"\n{'='*60}")
        print(f"RQ2: 序列类型对比 - {attack_type}")
        print(f"{'='*60}")
        
        results = {}
        
        for seq_type in self.config.sequence_types:
            print(f"\n测试: {seq_type} 序列")
            
            # 准备数据
            data = self.prepare_data(attack_type, seq_type)
            
            # 计算统计信息
            lengths = [len(d) for d in data]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            
            # 模拟生成效率测试 (如果有模型)
            model_path = get_model_path(attack_type, "rl", self.config.models_dir)
            
            if os.path.exists(model_path):
                model, tokenizer = self.runner.load_model(model_path)
                efficiency = self.measure_generation_efficiency(
                    model, tokenizer,
                    num_samples=min(self.config.num_payloads, 1000)
                )
            else:
                efficiency = {
                    "total_samples": 0,
                    "elapsed_time": 0,
                    "tokens_per_second": 0,
                }
            
            results[seq_type] = {
                "data_count": len(data),
                "avg_length_chars": avg_length,
                "efficiency": efficiency,
                "sample_payloads": data[:5],
            }
            
            print(f"  数据量: {len(data)}")
            print(f"  平均长度(字符): {avg_length:.1f}")
            if efficiency["elapsed_time"] > 0:
                print(f"  生成速度: {efficiency['tokens_per_second']:.1f} tokens/s")
        
        return results
    
    def run_all(self):
        """运行所有实验"""
        print("\n" + "="*80)
        print("RQ2: 攻击语法影响实验")
        print("="*80)
        
        for attack_type in self.config.attack_types:
            result = self.run_comparison(attack_type)
            self.results[attack_type] = result
            
            # 保存结果
            filename = f"{attack_type}_grammar_comparison.json"
            self.runner.save_results({
                "experiment": "rq2_grammar",
                "attack_type": attack_type,
                "timestamp": datetime.now().isoformat(),
                "results": result,
            }, filename)
        
        # 保存汇总
        self.runner.save_results({
            "experiment": "rq2_grammar",
            "timestamp": datetime.now().isoformat(),
            "all_results": self.results,
        }, "summary.json")
        
        print("\n" + "="*80)
        print("实验完成! 结果已保存")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RQ2: 攻击语法影响实验")
    
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--attack_types", type=str, nargs="+",
                        default=["sqli", "xss", "rce"])
    parser.add_argument("--num_payloads", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = RQ2Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        attack_types=args.attack_types,
        num_payloads=args.num_payloads,
        seed=args.seed,
    )
    
    experiment = RQ2Experiment(config)
    experiment.run_all()


if __name__ == "__main__":
    main()
