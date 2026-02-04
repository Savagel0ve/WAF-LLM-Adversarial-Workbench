"""
RQ6: GPTFuzzer能否生成语法无法生成的新payload？
===============================================

实验目标:
验证GPTFuzzer是否能超越原始语法，生成"新"的绕过payload。

实验流程:
1. 使用训练好的RL模型生成500K payload
2. 对每个payload判断是否能由原语法生成
3. 识别"新"payload (语法外)
4. 测试新payload的绕过率和功能性

评估指标:
- 新payload数量
- 新payload绕过率
- 新payload功能性验证率
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
from tqdm import tqdm
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import RQ6Config, get_model_path, get_data_path
from experiments.experiment_runner import ExperimentRunner
from experiments.baselines.grammar_fuzzer import (
    create_sqli_grammar,
    create_xss_grammar,
    create_rce_grammar,
    GrammarFuzzer,
)


class PayloadNoveltyChecker:
    """Payload新颖性检查器"""
    
    def __init__(self, attack_type: str, reference_payloads: List[str]):
        """
        初始化检查器
        
        Args:
            attack_type: 攻击类型
            reference_payloads: 参考payload列表 (语法生成的)
        """
        self.attack_type = attack_type
        self.reference_set = set(reference_payloads)
        
        # 构建参考特征
        self._build_reference_features(reference_payloads)
    
    def _build_reference_features(self, payloads: List[str]):
        """构建参考特征集合"""
        # n-gram特征
        self.reference_bigrams: Set[str] = set()
        self.reference_trigrams: Set[str] = set()
        
        for payload in payloads:
            # 字符级bigram
            for i in range(len(payload) - 1):
                self.reference_bigrams.add(payload[i:i+2])
            # 字符级trigram
            for i in range(len(payload) - 2):
                self.reference_trigrams.add(payload[i:i+3])
        
        # 关键模式
        self.reference_patterns = self._extract_patterns(payloads)
    
    def _extract_patterns(self, payloads: List[str]) -> Set[str]:
        """提取关键模式"""
        patterns = set()
        
        if self.attack_type == "sqli":
            # SQLi关键模式
            pattern_regexes = [
                r"UNION\s+SELECT",
                r"OR\s+[\d'\"]+\s*=\s*[\d'\"]+",
                r"AND\s+[\d'\"]+\s*=\s*[\d'\"]+",
                r"--\s*$",
                r"#\s*$",
                r"/\*.*\*/",
            ]
        elif self.attack_type == "xss":
            # XSS关键模式
            pattern_regexes = [
                r"<script[^>]*>",
                r"on\w+\s*=",
                r"javascript:",
                r"alert\s*\(",
            ]
        elif self.attack_type == "rce":
            # RCE关键模式
            pattern_regexes = [
                r";\s*\w+",
                r"\|\s*\w+",
                r"\$\([^)]+\)",
                r"`[^`]+`",
            ]
        else:
            pattern_regexes = []
        
        for payload in payloads:
            for regex in pattern_regexes:
                matches = re.findall(regex, payload, re.IGNORECASE)
                patterns.update(matches)
        
        return patterns
    
    def is_novel(self, payload: str) -> bool:
        """
        判断payload是否为新颖的
        
        判定条件:
        1. 不在参考集合中
        2. 包含新的n-gram组合
        
        Args:
            payload: 待检查的payload
            
        Returns:
            是否新颖
        """
        # 检查是否完全相同
        if payload in self.reference_set:
            return False
        
        # 检查是否有新的trigram
        novel_trigrams = 0
        for i in range(len(payload) - 2):
            trigram = payload[i:i+3]
            if trigram not in self.reference_trigrams:
                novel_trigrams += 1
        
        # 如果有超过20%的新trigram，认为是新颖的
        total_trigrams = max(len(payload) - 2, 1)
        novelty_ratio = novel_trigrams / total_trigrams
        
        return novelty_ratio > 0.2
    
    def get_novelty_score(self, payload: str) -> float:
        """
        计算payload的新颖度分数
        
        Args:
            payload: 待评估的payload
            
        Returns:
            新颖度分数 [0, 1]
        """
        if payload in self.reference_set:
            return 0.0
        
        # 基于新n-gram比例
        novel_bigrams = 0
        for i in range(len(payload) - 1):
            if payload[i:i+2] not in self.reference_bigrams:
                novel_bigrams += 1
        
        novel_trigrams = 0
        for i in range(len(payload) - 2):
            if payload[i:i+3] not in self.reference_trigrams:
                novel_trigrams += 1
        
        total_bigrams = max(len(payload) - 1, 1)
        total_trigrams = max(len(payload) - 2, 1)
        
        bigram_novelty = novel_bigrams / total_bigrams
        trigram_novelty = novel_trigrams / total_trigrams
        
        return (bigram_novelty + trigram_novelty) / 2


class RQ6Experiment:
    """RQ6实验: 超越语法的能力"""
    
    def __init__(self, config: RQ6Config):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.runner = ExperimentRunner(config)
        self.results: Dict[str, Dict] = {}
    
    def generate_grammar_reference(
        self,
        attack_type: str,
        num_samples: int = 100000
    ) -> List[str]:
        """
        使用语法生成参考payload集合
        
        Args:
            attack_type: 攻击类型
            num_samples: 生成数量
            
        Returns:
            参考payload列表
        """
        print(f"  生成语法参考集 ({num_samples} samples)...")
        
        grammar_creators = {
            "sqli": create_sqli_grammar,
            "xss": create_xss_grammar,
            "rce": create_rce_grammar,
        }
        
        grammar = grammar_creators[attack_type]()
        fuzzer = GrammarFuzzer(grammar, seed=self.config.seed)
        
        reference = fuzzer.generate_unique(num_samples, max_attempts=20)
        print(f"  生成了 {len(reference)} 个唯一参考payload")
        
        return reference
    
    def generate_model_payloads(
        self,
        attack_type: str,
        num_samples: int
    ) -> List[str]:
        """
        使用RL模型生成payload
        
        Args:
            attack_type: 攻击类型
            num_samples: 生成数量
            
        Returns:
            生成的payload列表
        """
        model_path = get_model_path(attack_type, "rl", self.config.models_dir)
        
        if not os.path.exists(model_path):
            print(f"  警告: RL模型不存在 {model_path}")
            # 回退到预训练模型
            model_path = get_model_path(attack_type, "pretrain", self.config.models_dir)
            if not os.path.exists(model_path):
                print(f"  警告: 预训练模型也不存在")
                return []
        
        print(f"  加载模型: {model_path}")
        model, tokenizer = self.runner.load_model(model_path)
        
        print(f"  生成 {num_samples} 个payload...")
        payloads = self.runner.generate_payloads(
            model, tokenizer,
            num_samples=num_samples,
            max_length=128,
            temperature=1.0,
            batch_size=100,
        )
        
        return payloads
    
    def analyze_novelty(
        self,
        generated: List[str],
        reference: List[str],
        attack_type: str
    ) -> Dict:
        """
        分析生成payload的新颖性
        
        Args:
            generated: 生成的payload
            reference: 参考payload
            attack_type: 攻击类型
            
        Returns:
            分析结果
        """
        print(f"  分析新颖性...")
        
        checker = PayloadNoveltyChecker(attack_type, reference)
        
        novel_payloads = []
        novelty_scores = []
        
        for payload in tqdm(generated, desc="检查新颖性"):
            is_novel = checker.is_novel(payload)
            score = checker.get_novelty_score(payload)
            
            if is_novel:
                novel_payloads.append(payload)
            novelty_scores.append(score)
        
        return {
            "total_generated": len(generated),
            "unique_generated": len(set(generated)),
            "novel_count": len(novel_payloads),
            "novel_ratio": len(novel_payloads) / len(generated) if generated else 0,
            "avg_novelty_score": sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0,
            "novel_payloads": novel_payloads,
        }
    
    def test_novel_payloads(
        self,
        novel_payloads: List[str],
        attack_type: str
    ) -> Dict:
        """
        测试新颖payload的WAF绕过和功能性
        
        Args:
            novel_payloads: 新颖的payload列表
            attack_type: 攻击类型
            
        Returns:
            测试结果
        """
        if not novel_payloads:
            return {
                "tested": 0,
                "bypassed": 0,
                "bypass_rate": 0,
            }
        
        print(f"  测试 {len(novel_payloads)} 个新颖payload...")
        
        waf_url = self.config.waf_urls.get("modsecurity", "http://localhost:8001")
        
        # 限制测试数量
        test_payloads = novel_payloads[:min(len(novel_payloads), 1000)]
        
        bypassed, blocked, errors = self.runner.test_waf(
            test_payloads, waf_url
        )
        
        return {
            "tested": len(test_payloads),
            "bypassed": len(bypassed),
            "blocked": len(blocked),
            "errors": len(errors),
            "bypass_rate": len(bypassed) / len(test_payloads) if test_payloads else 0,
            "bypassed_samples": bypassed[:20],
        }
    
    def run_novelty_experiment(self, attack_type: str) -> Dict:
        """
        运行单个攻击类型的新颖性实验
        
        Args:
            attack_type: 攻击类型
            
        Returns:
            实验结果
        """
        print(f"\n{'='*60}")
        print(f"RQ6: 新颖性实验 - {attack_type}")
        print(f"{'='*60}")
        
        # 1. 生成语法参考集
        reference = self.generate_grammar_reference(attack_type, num_samples=50000)
        
        # 2. 生成模型payload
        generated = self.generate_model_payloads(attack_type, self.config.num_generate)
        
        if not generated:
            return {
                "attack_type": attack_type,
                "error": "Failed to generate payloads",
            }
        
        # 3. 分析新颖性
        novelty_analysis = self.analyze_novelty(generated, reference, attack_type)
        
        # 4. 测试新颖payload
        novel_payloads = novelty_analysis["novel_payloads"]
        test_results = self.test_novel_payloads(novel_payloads, attack_type)
        
        # 汇总结果
        result = {
            "attack_type": attack_type,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_generate": self.config.num_generate,
                "reference_size": len(reference),
            },
            "novelty_analysis": {
                "total_generated": novelty_analysis["total_generated"],
                "unique_generated": novelty_analysis["unique_generated"],
                "novel_count": novelty_analysis["novel_count"],
                "novel_ratio": novelty_analysis["novel_ratio"],
                "avg_novelty_score": novelty_analysis["avg_novelty_score"],
            },
            "waf_test": test_results,
        }
        
        # 打印结果
        print(f"\n结果:")
        print(f"  生成总数: {novelty_analysis['total_generated']}")
        print(f"  唯一数: {novelty_analysis['unique_generated']}")
        print(f"  新颖数: {novelty_analysis['novel_count']}")
        print(f"  新颖比例: {novelty_analysis['novel_ratio']:.4f}")
        print(f"  新颖payload绕过率: {test_results['bypass_rate']:.4f}")
        
        return result
    
    def run_all(self):
        """运行所有实验"""
        print("\n" + "="*80)
        print("RQ6: 超越语法能力实验")
        print("="*80)
        
        for attack_type in self.config.attack_types:
            result = self.run_novelty_experiment(attack_type)
            self.results[attack_type] = result
            
            # 保存结果
            self.runner.save_results({
                "experiment": "rq6_novel_payloads",
                **result
            }, f"{attack_type}_novelty.json")
        
        # 保存汇总
        self._save_summary()
    
    def _save_summary(self):
        """保存汇总结果"""
        summary = {
            "experiment": "rq6_novel_payloads",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_generate": self.config.num_generate,
                "attack_types": self.config.attack_types,
            },
            "results": {}
        }
        
        for attack_type, result in self.results.items():
            if "error" not in result:
                summary["results"][attack_type] = {
                    "novel_count": result["novelty_analysis"]["novel_count"],
                    "novel_ratio": result["novelty_analysis"]["novel_ratio"],
                    "novel_bypass_rate": result["waf_test"]["bypass_rate"],
                }
        
        self.runner.save_results(summary, "summary.json")
        
        print("\n" + "="*80)
        print("实验完成! 结果已保存")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="RQ6: 超越语法能力实验")
    
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--attack_types", type=str, nargs="+",
                        default=["sqli"])  # 只测试SQLi
    parser.add_argument("--num_generate", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = RQ6Config(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        attack_types=args.attack_types,
        num_generate=args.num_generate,
        seed=args.seed,
    )
    
    experiment = RQ6Experiment(config)
    experiment.run_all()


if __name__ == "__main__":
    main()
