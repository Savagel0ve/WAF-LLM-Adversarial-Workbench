"""
Random Fuzzer 基线方法
======================

随机从数据集中采样或基于简单规则随机生成payload。
这是论文中最简单的基线方法。
"""

import random
import string
from typing import List, Dict, Optional, Set
from pathlib import Path


class RandomFuzzer:
    """
    随机Fuzzer基线
    
    两种模式:
    1. 从已有数据集中随机采样
    2. 基于简单规则随机生成
    """
    
    def __init__(
        self,
        attack_type: str = "sqli",
        data_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        初始化随机Fuzzer
        
        Args:
            attack_type: 攻击类型 (sqli/xss/rce)
            data_path: 数据文件路径 (用于采样模式)
            seed: 随机种子
        """
        self.attack_type = attack_type
        self.seed = seed
        random.seed(seed)
        
        # 加载数据集 (如果提供)
        self.payloads: List[str] = []
        if data_path and Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                self.payloads = [line.strip() for line in f if line.strip()]
            print(f"加载 {len(self.payloads)} 条payload用于采样")
        
        # 生成组件
        self._init_components()
    
    def _init_components(self):
        """初始化各攻击类型的生成组件"""
        
        # SQLi 组件
        self.sqli_components = {
            "contexts": ["", "'", "\"", "1", "0", "-1"],
            "operators": ["AND", "OR", "NOT", "XOR"],
            "comparisons": ["=", "!=", "<", ">", "<=", ">=", "LIKE"],
            "functions": ["SELECT", "UNION", "INSERT", "UPDATE", "DELETE", "DROP"],
            "keywords": ["FROM", "WHERE", "ORDER BY", "GROUP BY", "HAVING", "LIMIT"],
            "comments": ["--", "#", "/**/", "-- -", ";--"],
            "encodings": ["%20", "%0a", "%0d", "%09", "/**/"],
            "booleans": ["1=1", "1=0", "'a'='a'", "TRUE", "FALSE"],
            "numbers": ["1", "0", "-1", "999", "NULL"],
        }
        
        # XSS 组件
        self.xss_components = {
            "tags": ["<script>", "<img", "<svg", "<body", "<iframe", "<input"],
            "events": ["onerror", "onload", "onclick", "onmouseover", "onfocus"],
            "payloads": ["alert(1)", "alert('XSS')", "confirm(1)", "prompt(1)"],
            "closers": [">", "/>", "</script>", "'>", "\">"],
            "encodings": ["&#x", "&#", "%3C", "%3E", "\\x", "\\u"],
        }
        
        # RCE 组件
        self.rce_components = {
            "separators": [";", "|", "||", "&&", "&", "\n", "`"],
            "commands": ["ls", "cat", "id", "whoami", "pwd", "uname", "echo"],
            "flags": ["-la", "-a", "-l", "-h", "-r", "-f"],
            "paths": ["/etc/passwd", "/etc/shadow", "../", "../../", "/tmp/"],
            "encodings": ["$IFS", "${IFS}", "%20", "%0a", "%0d"],
        }
    
    def generate_random_sqli(self) -> str:
        """生成随机SQLi payload"""
        parts = []
        
        # 上下文
        parts.append(random.choice(self.sqli_components["contexts"]))
        
        # 随机组合
        if random.random() > 0.5:
            parts.append(random.choice(self.sqli_components["operators"]))
        
        if random.random() > 0.3:
            parts.append(random.choice(self.sqli_components["booleans"]))
        
        if random.random() > 0.5:
            parts.append(random.choice(self.sqli_components["functions"]))
            parts.append(random.choice(self.sqli_components["keywords"]))
        
        # 注释
        if random.random() > 0.3:
            parts.append(random.choice(self.sqli_components["comments"]))
        
        # 随机使用编码
        sep = random.choice(self.sqli_components["encodings"] + [" "])
        return sep.join(parts)
    
    def generate_random_xss(self) -> str:
        """生成随机XSS payload"""
        parts = []
        
        # 标签
        tag = random.choice(self.xss_components["tags"])
        parts.append(tag)
        
        # 事件处理器
        if random.random() > 0.3:
            event = random.choice(self.xss_components["events"])
            payload = random.choice(self.xss_components["payloads"])
            parts.append(f'{event}="{payload}"')
        
        # 关闭标签
        parts.append(random.choice(self.xss_components["closers"]))
        
        return " ".join(parts)
    
    def generate_random_rce(self) -> str:
        """生成随机RCE payload"""
        parts = []
        
        # 分隔符
        sep = random.choice(self.rce_components["separators"])
        
        # 命令
        cmd = random.choice(self.rce_components["commands"])
        
        # 标志
        if random.random() > 0.5:
            cmd += " " + random.choice(self.rce_components["flags"])
        
        # 路径
        if random.random() > 0.5:
            cmd += " " + random.choice(self.rce_components["paths"])
        
        return sep + cmd
    
    def generate(self, num_samples: int = 1) -> List[str]:
        """
        生成payload
        
        如果有数据集，则从中采样；否则随机生成。
        
        Args:
            num_samples: 生成数量
            
        Returns:
            payload列表
        """
        # 如果有数据集，优先采样
        if self.payloads:
            if num_samples <= len(self.payloads):
                return random.sample(self.payloads, num_samples)
            else:
                # 数据集不够，循环采样
                return [random.choice(self.payloads) for _ in range(num_samples)]
        
        # 随机生成
        generators = {
            "sqli": self.generate_random_sqli,
            "xss": self.generate_random_xss,
            "rce": self.generate_random_rce,
        }
        
        generator = generators.get(self.attack_type, self.generate_random_sqli)
        return [generator() for _ in range(num_samples)]
    
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
            seen.add(payload)
            attempts += 1
        
        return list(seen)[:num_samples]


class RandomFuzzerWithMutation(RandomFuzzer):
    """
    带变异的随机Fuzzer
    
    在随机生成的基础上增加变异操作。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 变异操作
        self.mutations = [
            self._case_mutation,
            self._encoding_mutation,
            self._space_mutation,
            self._duplicate_mutation,
            self._insert_mutation,
        ]
    
    def _case_mutation(self, payload: str) -> str:
        """大小写变异"""
        result = []
        for c in payload:
            if random.random() > 0.5:
                result.append(c.swapcase())
            else:
                result.append(c)
        return ''.join(result)
    
    def _encoding_mutation(self, payload: str) -> str:
        """URL编码变异"""
        result = []
        for c in payload:
            if random.random() > 0.8 and c.isalpha():
                result.append(f'%{ord(c):02X}')
            else:
                result.append(c)
        return ''.join(result)
    
    def _space_mutation(self, payload: str) -> str:
        """空白字符变异"""
        replacements = [' ', '%20', '%09', '%0a', '/**/']
        return payload.replace(' ', random.choice(replacements))
    
    def _duplicate_mutation(self, payload: str) -> str:
        """重复字符变异"""
        if len(payload) < 2:
            return payload
        idx = random.randint(0, len(payload) - 1)
        return payload[:idx] + payload[idx] * random.randint(1, 3) + payload[idx:]
    
    def _insert_mutation(self, payload: str) -> str:
        """插入注释变异"""
        if len(payload) < 2:
            return payload
        idx = random.randint(1, len(payload) - 1)
        insert = random.choice(['/**/', '-- ', '# ', ''])
        return payload[:idx] + insert + payload[idx:]
    
    def mutate(self, payload: str, num_mutations: int = 1) -> str:
        """
        对payload进行变异
        
        Args:
            payload: 原始payload
            num_mutations: 变异次数
            
        Returns:
            变异后的payload
        """
        for _ in range(num_mutations):
            mutation = random.choice(self.mutations)
            payload = mutation(payload)
        return payload
    
    def generate(self, num_samples: int = 1) -> List[str]:
        """生成带变异的payload"""
        base_payloads = super().generate(num_samples)
        
        # 对部分进行变异
        results = []
        for payload in base_payloads:
            if random.random() > 0.5:
                payload = self.mutate(payload, random.randint(1, 3))
            results.append(payload)
        
        return results


if __name__ == "__main__":
    # 测试
    print("测试 RandomFuzzer:")
    
    for attack_type in ["sqli", "xss", "rce"]:
        print(f"\n{attack_type.upper()}:")
        fuzzer = RandomFuzzer(attack_type=attack_type)
        samples = fuzzer.generate(5)
        for i, s in enumerate(samples):
            print(f"  [{i+1}] {s}")
    
    print("\n\n测试 RandomFuzzerWithMutation:")
    fuzzer = RandomFuzzerWithMutation(attack_type="sqli")
    samples = fuzzer.generate(5)
    for i, s in enumerate(samples):
        print(f"  [{i+1}] {s}")
