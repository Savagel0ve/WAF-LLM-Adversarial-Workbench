"""
奖励模型 - 改进的多维度奖励函数

支持两种奖励模型:
1. NeuralRewardModel - 基于训练的GPT-2分类模型（推荐）
2. RewardModel - 基于规则的多维度奖励函数
3. SimpleRewardModel - 简单的二值奖励
"""
from typing import Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


# 尝试导入神经网络模型
try:
    from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
    NEURAL_MODEL_AVAILABLE = True
except ImportError:
    NEURAL_MODEL_AVAILABLE = False


@dataclass
class RewardComponents:
    """奖励组成部分"""
    waf_bypass: float = 0.0      # WAF绕过奖励
    syntax_valid: float = 0.0     # 语法正确性奖励
    executable: float = 0.0       # 可执行性奖励
    novelty: float = 0.0          # 新颖性奖励
    length_penalty: float = 0.0   # 长度惩罚
    total: float = 0.0            # 总奖励
    
    def to_dict(self) -> Dict:
        return {
            "waf_bypass": self.waf_bypass,
            "syntax_valid": self.syntax_valid,
            "executable": self.executable,
            "novelty": self.novelty,
            "length_penalty": self.length_penalty,
            "total": self.total
        }


class RewardModel:
    """
    改进的奖励模型 - 多维度奖励函数
    
    相比GPTFuzzer的简单二值奖励,本模型提供:
    1. 多维度奖励: 不仅看WAF绕过,还考虑语法和执行性
    2. 中间奖励: 给予"接近成功"的样本部分奖励
    3. 新颖性奖励: 鼓励生成多样化的payload
    """
    
    def __init__(self, 
                 w_bypass: float = 10.0,
                 w_syntax: float = 2.0,
                 w_executable: float = 5.0,
                 w_novelty: float = 1.0,
                 w_length_penalty: float = 0.1,
                 seen_payloads: Optional[set] = None):
        """
        初始化奖励模型
        
        Args:
            w_bypass: WAF绕过奖励权重
            w_syntax: 语法正确性奖励权重
            w_executable: 可执行性奖励权重
            w_novelty: 新颖性奖励权重
            w_length_penalty: 长度惩罚权重
            seen_payloads: 已见过的payload集合(用于新颖性检查)
        """
        self.w_bypass = w_bypass
        self.w_syntax = w_syntax
        self.w_executable = w_executable
        self.w_novelty = w_novelty
        self.w_length_penalty = w_length_penalty
        
        self.seen_payloads = seen_payloads if seen_payloads is not None else set()
        
        # 统计
        self.reward_history = []
    
    def compute_reward(self, 
                      payload: str,
                      waf_response: Dict,
                      verifier_result: Optional[Dict] = None) -> RewardComponents:
        """
        计算payload的奖励
        
        Args:
            payload: 生成的payload
            waf_response: WAF响应 (包含blocked, status_code等)
            verifier_result: 验证器结果 (包含syntax_valid, executable等)
            
        Returns:
            RewardComponents对象
        """
        reward = RewardComponents()
        
        # R1: WAF绕过奖励
        if not waf_response.get("blocked", True):
            reward.waf_bypass = self.w_bypass  # 成功绕过
        else:
            # 根据WAF响应给予中间奖励
            status_code = waf_response.get("status_code", 403)
            if status_code == 200:
                reward.waf_bypass = self.w_bypass * 0.2  # 部分绕过
            elif status_code == 403:
                reward.waf_bypass = -self.w_bypass * 0.1  # 完全拦截
            else:
                reward.waf_bypass = 0.0
        
        # R2: 语法正确性奖励
        if verifier_result:
            if verifier_result.get("syntax_valid", False):
                reward.syntax_valid = self.w_syntax
            else:
                reward.syntax_valid = -self.w_syntax * 0.5  # 惩罚无效语法
        
        # R3: 可执行性奖励
        if verifier_result:
            if verifier_result.get("executable", False):
                reward.executable = self.w_executable
        
        # R4: 新颖性奖励
        if payload not in self.seen_payloads:
            reward.novelty = self.w_novelty
            self.seen_payloads.add(payload)
        else:
            reward.novelty = -self.w_novelty * 0.3  # 惩罚重复
        
        # R5: 长度惩罚 (避免生成过长的payload)
        length_ratio = len(payload) / 200.0  # 假设200是理想长度
        if length_ratio > 1.5:
            reward.length_penalty = -self.w_length_penalty * (length_ratio - 1.5)
        
        # 计算总奖励
        reward.total = (
            reward.waf_bypass +
            reward.syntax_valid +
            reward.executable +
            reward.novelty +
            reward.length_penalty
        )
        
        # 记录
        self.reward_history.append(reward.total)
        
        return reward
    
    def compute_batch_rewards(self,
                             payloads: list,
                             waf_responses: list,
                             verifier_results: Optional[list] = None) -> list:
        """
        批量计算奖励
        
        Args:
            payloads: payload列表
            waf_responses: WAF响应列表
            verifier_results: 验证器结果列表
            
        Returns:
            奖励列表(float)
        """
        if verifier_results is None:
            verifier_results = [None] * len(payloads)
        
        rewards = []
        for payload, waf_resp, verifier_resp in zip(payloads, waf_responses, verifier_results):
            reward_components = self.compute_reward(payload, waf_resp, verifier_resp)
            rewards.append(reward_components.total)
        
        return rewards
    
    def get_statistics(self) -> Dict:
        """获取奖励统计信息"""
        if not self.reward_history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }
        
        history = np.array(self.reward_history)
        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
            "count": len(history)
        }
    
    def reset(self):
        """重置奖励模型"""
        self.reward_history = []
        self.seen_payloads = set()


class SimpleRewardModel:
    """
    简单奖励模型 - 仅基于WAF绕过
    用于对照实验
    """
    
    def __init__(self, reward_value: float = 1.0):
        self.reward_value = reward_value
        self.reward_history = []
    
    def compute_reward(self, payload: str, waf_response: Dict, verifier_result: Optional[Dict] = None) -> float:
        """计算简单奖励"""
        if waf_response.get("blocked", True):
            reward = -self.reward_value
        else:
            reward = self.reward_value
        
        self.reward_history.append(reward)
        return reward
    
    def compute_batch_rewards(self, payloads: list, waf_responses: list, verifier_results: Optional[list] = None) -> list:
        """批量计算奖励"""
        return [self.compute_reward(p, r) for p, r in zip(payloads, waf_responses)]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.reward_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        history = np.array(self.reward_history)
        return {
            "mean": float(np.mean(history)),
            "std": float(np.std(history)),
            "min": float(np.min(history)),
            "max": float(np.max(history)),
            "count": len(history)
        }


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("奖励模型测试")
    print("="*60)
    
    # 创建奖励模型
    reward_model = RewardModel()
    
    # 测试案例
    test_cases = [
        {
            "payload": "' OR 1=1 --",
            "waf_response": {"blocked": False, "status_code": 200},
            "verifier_result": {"syntax_valid": True, "executable": True}
        },
        {
            "payload": "UNION SELECT",
            "waf_response": {"blocked": True, "status_code": 403},
            "verifier_result": {"syntax_valid": False, "executable": False}
        },
        {
            "payload": "1' AND 1=1 --",
            "waf_response": {"blocked": False, "status_code": 200},
            "verifier_result": {"syntax_valid": True, "executable": True}
        },
    ]
    
    print("\n测试案例:")
    for i, case in enumerate(test_cases, 1):
        reward_comp = reward_model.compute_reward(
            case["payload"],
            case["waf_response"],
            case["verifier_result"]
        )
        
        print(f"\n案例 {i}: {case['payload']}")
        print(f"  WAF绕过: {reward_comp.waf_bypass:+.2f}")
        print(f"  语法正确: {reward_comp.syntax_valid:+.2f}")
        print(f"  可执行: {reward_comp.executable:+.2f}")
        print(f"  新颖性: {reward_comp.novelty:+.2f}")
        print(f"  总奖励: {reward_comp.total:+.2f}")
    
    # 统计
    print("\n" + "="*60)
    print("统计信息:")
    stats = reward_model.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
