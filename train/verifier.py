"""
智能验证器 - 验证payload的语法正确性和可执行性
"""
import re
from typing import Dict, Optional
from pathlib import Path
import logging


class PayloadVerifier:
    """Payload验证器基类"""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
        self.logger = logging.getLogger(__name__)
    
    def verify(self, payload: str) -> Dict:
        """
        验证payload
        
        Returns:
            {
                "syntax_valid": bool,
                "executable": bool,
                "is_novel": bool,
                "error": Optional[str]
            }
        """
        raise NotImplementedError


class SQLiVerifier(PayloadVerifier):
    """SQL注入验证器"""
    
    def __init__(self):
        super().__init__("sqli")
        
        # SQL关键词
        self.sql_keywords = [
            "select", "union", "insert", "update", "delete", "drop",
            "create", "alter", "where", "from", "and", "or", "order",
            "by", "group", "having", "join", "on", "as", "like"
        ]
        
        # SQL函数
        self.sql_functions = [
            "count", "sum", "avg", "max", "min", "concat", "substring",
            "version", "database", "user", "sleep", "benchmark"
        ]
    
    def verify(self, payload: str) -> Dict:
        """验证SQL注入payload"""
        result = {
            "syntax_valid": False,
            "executable": False,
            "is_novel": True,
            "error": None
        }
        
        try:
            # 基本语法检查
            result["syntax_valid"] = self._check_syntax(payload)
            
            # 可执行性检查(简化版)
            if result["syntax_valid"]:
                result["executable"] = self._check_executable(payload)
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _check_syntax(self, payload: str) -> bool:
        """检查SQL语法"""
        payload_lower = payload.lower()
        
        # 检查是否包含SQL关键词或函数
        has_keyword = any(kw in payload_lower for kw in self.sql_keywords)
        has_function = any(fn in payload_lower for fn in self.sql_functions)
        
        if not (has_keyword or has_function):
            return False
        
        # 检查括号匹配
        if payload.count('(') != payload.count(')'):
            return False
        
        # 检查引号匹配(简化)
        single_quotes = payload.count("'")
        double_quotes = payload.count('"')
        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            # 允许SQL注释闭合
            if '--' not in payload and '#' not in payload:
                return False
        
        return True
    
    def _check_executable(self, payload: str) -> bool:
        """检查是否可能执行"""
        payload_lower = payload.lower()
        
        # 常见的可执行SQL注入模式
        executable_patterns = [
            r"union\s+select",
            r"'\s*or\s+",
            r"'\s*and\s+",
            r";\s*(select|insert|update|delete|drop)",
            r"sleep\s*\(",
            r"benchmark\s*\(",
        ]
        
        for pattern in executable_patterns:
            if re.search(pattern, payload_lower):
                return True
        
        return False


class XSSVerifier(PayloadVerifier):
    """XSS验证器"""
    
    def __init__(self):
        super().__init__("xss")
        
        # XSS标签和事件
        self.xss_tags = ["script", "img", "svg", "iframe", "object", "embed"]
        self.xss_events = [
            "onerror", "onload", "onclick", "onmouseover",
            "onfocus", "onblur", "oninput", "onchange"
        ]
        self.xss_protocols = ["javascript:", "data:", "vbscript:"]
    
    def verify(self, payload: str) -> Dict:
        """验证XSS payload"""
        result = {
            "syntax_valid": False,
            "executable": False,
            "is_novel": True,
            "error": None
        }
        
        try:
            result["syntax_valid"] = self._check_syntax(payload)
            if result["syntax_valid"]:
                result["executable"] = self._check_executable(payload)
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _check_syntax(self, payload: str) -> bool:
        """检查XSS语法"""
        payload_lower = payload.lower()
        
        # 检查是否包含XSS相关元素
        has_tag = any(f"<{tag}" in payload_lower for tag in self.xss_tags)
        has_event = any(event in payload_lower for event in self.xss_events)
        has_protocol = any(proto in payload_lower for proto in self.xss_protocols)
        
        if not (has_tag or has_event or has_protocol):
            return False
        
        # 检查HTML标签匹配(简化)
        open_tags = payload_lower.count('<')
        close_tags = payload_lower.count('>')
        if open_tags > 0 and close_tags == 0:
            return False
        
        return True
    
    def _check_executable(self, payload: str) -> bool:
        """检查是否可能执行"""
        payload_lower = payload.lower()
        
        # 常见的可执行XSS模式
        executable_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<img[^>]+src\s*=",
            r"<svg[^>]*>",
        ]
        
        for pattern in executable_patterns:
            if re.search(pattern, payload_lower):
                return True
        
        return False


class RCEVerifier(PayloadVerifier):
    """RCE(远程代码执行)验证器"""
    
    def __init__(self):
        super().__init__("rce")
        
        # RCE命令和函数
        self.rce_commands = [
            "ls", "cat", "wget", "curl", "nc", "bash", "sh",
            "python", "perl", "php", "ruby", "node"
        ]
        self.rce_functions = [
            "eval", "exec", "system", "popen", "shell_exec",
            "passthru", "proc_open"
        ]
    
    def verify(self, payload: str) -> Dict:
        """验证RCE payload"""
        result = {
            "syntax_valid": False,
            "executable": False,
            "is_novel": True,
            "error": None
        }
        
        try:
            result["syntax_valid"] = self._check_syntax(payload)
            if result["syntax_valid"]:
                result["executable"] = self._check_executable(payload)
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _check_syntax(self, payload: str) -> bool:
        """检查RCE语法"""
        payload_lower = payload.lower()
        
        # 检查命令注入符号
        injection_chars = [';', '|', '&', '$', '`', '(', ')', '{', '}']
        has_injection_char = any(char in payload for char in injection_chars)
        
        # 检查命令或函数
        has_command = any(cmd in payload_lower for cmd in self.rce_commands)
        has_function = any(func in payload_lower for func in self.rce_functions)
        
        return has_injection_char and (has_command or has_function)
    
    def _check_executable(self, payload: str) -> bool:
        """检查是否可能执行"""
        payload_lower = payload.lower()
        
        # 常见的可执行RCE模式
        executable_patterns = [
            r";\s*\w+",  # 命令分隔
            r"\|\s*\w+",  # 管道
            r"\$\(.+\)",  # 命令替换
            r"`[^`]+`",  # 反引号命令替换
            r"eval\s*\(",  # eval函数
        ]
        
        for pattern in executable_patterns:
            if re.search(pattern, payload_lower):
                return True
        
        return False


class UniversalVerifier:
    """通用验证器 - 根据攻击类型选择合适的验证器"""
    
    def __init__(self):
        self.verifiers = {
            "sqli": SQLiVerifier(),
            "xss": XSSVerifier(),
            "rce": RCEVerifier()
        }
        self.seen_payloads = set()
    
    def verify(self, payload: str, attack_type: str) -> Dict:
        """
        验证payload
        
        Args:
            payload: 待验证的payload
            attack_type: 攻击类型 ("sqli", "xss", "rce")
            
        Returns:
            验证结果字典
        """
        verifier = self.verifiers.get(attack_type)
        if not verifier:
            return {
                "syntax_valid": False,
                "executable": False,
                "is_novel": False,
                "error": f"Unknown attack type: {attack_type}"
            }
        
        result = verifier.verify(payload)
        
        # 检查新颖性
        result["is_novel"] = payload not in self.seen_payloads
        if result["is_novel"]:
            self.seen_payloads.add(payload)
        
        return result
    
    def reset(self):
        """重置seen payloads"""
        self.seen_payloads = set()


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("验证器测试")
    print("="*60)
    
    verifier = UniversalVerifier()
    
    # 测试SQLi
    print("\n【SQLi验证】")
    sqli_tests = [
        "' OR 1=1 --",
        "UNION SELECT * FROM users",
        "normal text",
        "'; DROP TABLE users--"
    ]
    
    for payload in sqli_tests:
        result = verifier.verify(payload, "sqli")
        status = "✅" if result["syntax_valid"] else "❌"
        exec_status = "🔥" if result["executable"] else "  "
        print(f"{status} {exec_status} {payload[:40]}")
        print(f"   语法: {result['syntax_valid']}, 可执行: {result['executable']}")
    
    # 测试XSS
    print("\n【XSS验证】")
    xss_tests = [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "normal text",
        "javascript:alert(1)"
    ]
    
    for payload in xss_tests:
        result = verifier.verify(payload, "xss")
        status = "✅" if result["syntax_valid"] else "❌"
        exec_status = "🔥" if result["executable"] else "  "
        print(f"{status} {exec_status} {payload[:40]}")
        print(f"   语法: {result['syntax_valid']}, 可执行: {result['executable']}")
    
    # 测试RCE
    print("\n【RCE验证】")
    rce_tests = [
        "; ls -la",
        "| cat /etc/passwd",
        "normal text",
        "$(whoami)"
    ]
    
    for payload in rce_tests:
        result = verifier.verify(payload, "rce")
        status = "✅" if result["syntax_valid"] else "❌"
        exec_status = "🔥" if result["executable"] else "  "
        print(f"{status} {exec_status} {payload[:40]}")
        print(f"   语法: {result['syntax_valid']}, 可执行: {result['executable']}")
