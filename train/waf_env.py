"""
WAF环境接口 - 用于与WAF交互和获取反馈
"""
import requests
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class WAFResponse:
    """WAF响应数据类"""
    blocked: bool
    status_code: int
    response_text: str
    response_time: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "blocked": self.blocked,
            "status_code": self.status_code,
            "response_length": len(self.response_text),
            "response_time": self.response_time,
            "error": self.error
        }


class WAFEnvironment:
    """WAF测试环境"""
    
    def __init__(self, 
                 waf_type: str = "modsecurity",
                 modsecurity_url: str = "http://localhost:8001",
                 naxsi_url: str = "http://localhost:8002",
                 timeout: int = 10,
                 max_retries: int = 3):
        """
        初始化WAF环境
        
        Args:
            waf_type: WAF类型 ("modsecurity" 或 "naxsi")
            modsecurity_url: ModSecurity WAF URL
            naxsi_url: Naxsi WAF URL
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        self.waf_type = waf_type.lower()
        self.modsecurity_url = modsecurity_url
        self.naxsi_url = naxsi_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 请求计数
        self.request_count = 0
        self.blocked_count = 0
        self.error_count = 0
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 当前WAF URL
        self.current_url = self.modsecurity_url if waf_type == "modsecurity" else self.naxsi_url
        
        self.logger.info(f"WAF环境初始化: {self.waf_type} @ {self.current_url}")
    
    def send_payload(self, 
                    payload: str, 
                    attack_type: str = "sqli",
                    method: str = "GET") -> WAFResponse:
        """
        发送payload到WAF
        
        Args:
            payload: 攻击payload
            attack_type: 攻击类型 ("sqli", "xss", "rce")
            method: HTTP方法
            
        Returns:
            WAFResponse对象
        """
        self.request_count += 1
        
        # 构造请求
        if method.upper() == "GET":
            params = {"input": payload, "type": attack_type}
            data = None
        else:
            params = None
            data = {"input": payload, "type": attack_type}
        
        # 发送请求(带重试)
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = requests.request(
                    method=method.upper(),
                    url=self.current_url,
                    params=params,
                    data=data,
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                # 判断是否被拦截
                blocked = self._is_blocked(response)
                
                if blocked:
                    self.blocked_count += 1
                
                return WAFResponse(
                    blocked=blocked,
                    status_code=response.status_code,
                    response_text=response.text[:1000],  # 限制长度
                    response_time=response_time
                )
                
            except requests.Timeout:
                self.error_count += 1
                if attempt == self.max_retries - 1:
                    return WAFResponse(
                        blocked=True,  # 超时视为拦截
                        status_code=0,
                        response_text="",
                        response_time=self.timeout,
                        error="Timeout"
                    )
                time.sleep(0.5 * (attempt + 1))  # 指数退避
                
            except requests.RequestException as e:
                self.error_count += 1
                if attempt == self.max_retries - 1:
                    return WAFResponse(
                        blocked=True,
                        status_code=0,
                        response_text="",
                        response_time=0,
                        error=str(e)
                    )
                time.sleep(0.5 * (attempt + 1))
        
        # 不应该到达这里
        return WAFResponse(
            blocked=True,
            status_code=0,
            response_text="",
            response_time=0,
            error="Max retries exceeded"
        )
    
    def _is_blocked(self, response: requests.Response) -> bool:
        """
        判断请求是否被WAF拦截
        
        Args:
            response: requests Response对象
            
        Returns:
            是否被拦截
        """
        # 常见的WAF拦截状态码
        blocked_status_codes = [403, 406, 419, 429, 503]
        
        if response.status_code in blocked_status_codes:
            return True
        
        # 状态码200通常表示通过
        if response.status_code == 200:
            # 但需要检查响应内容是否包含明确的拦截信息
            response_lower = response.text.lower()
            
            # 只检查明确的拦截消息（更严格的匹配）
            strict_blocked_keywords = [
                "request blocked",
                "access denied",
                "forbidden",
                "not acceptable",
                "this request has been blocked",
                "your request was denied",
                "modsecurity: access denied",
                "blocked by",
            ]
            
            for keyword in strict_blocked_keywords:
                if keyword in response_lower:
                    return True
            
            # 如果没有明确的拦截消息，认为是通过
            return False
        
        # 其他状态码（如500, 502等）也认为是拦截
        if response.status_code >= 500:
            return True
        
        return False
    
    def batch_send(self, payloads: List[str], attack_type: str = "sqli") -> List[WAFResponse]:
        """
        批量发送payload
        
        Args:
            payloads: payload列表
            attack_type: 攻击类型
            
        Returns:
            WAFResponse列表
        """
        responses = []
        for payload in payloads:
            response = self.send_payload(payload, attack_type)
            responses.append(response)
        return responses
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        bypass_count = self.request_count - self.blocked_count - self.error_count
        bypass_rate = (bypass_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "blocked": self.blocked_count,
            "bypassed": bypass_count,
            "errors": self.error_count,
            "bypass_rate": bypass_rate,
            "blocked_rate": (self.blocked_count / self.request_count * 100) if self.request_count > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.request_count = 0
        self.blocked_count = 0
        self.error_count = 0
    
    def test_connection(self) -> bool:
        """测试WAF连接"""
        try:
            response = requests.get(self.current_url, timeout=5)
            self.logger.info(f"WAF连接测试成功: {response.status_code}")
            return True
        except Exception as e:
            self.logger.error(f"WAF连接测试失败: {e}")
            return False


class MockWAFEnvironment(WAFEnvironment):
    """
    模拟WAF环境 - 用于本地测试
    不需要真实的WAF服务
    """
    
    def __init__(self, block_rate: float = 0.7):
        """
        Args:
            block_rate: 模拟的拦截率 (0.0-1.0)
        """
        super().__init__()
        self.block_rate = block_rate
        self.logger.info(f"使用模拟WAF环境 (拦截率: {block_rate*100}%)")
    
    def send_payload(self, payload: str, attack_type: str = "sqli", method: str = "GET") -> WAFResponse:
        """模拟发送payload"""
        self.request_count += 1
        
        # 模拟延迟
        time.sleep(0.01)
        
        # 简单的拦截规则(基于关键词)
        blocked_keywords = {
            "sqli": ["union", "select", "drop", "insert", "--", "/*"],
            "xss": ["<script>", "javascript:", "onerror", "onload"],
            "rce": [";", "|", "&", "$(", "`"]
        }
        
        blocked = False
        
        # 检查关键词
        payload_lower = payload.lower()
        for keyword in blocked_keywords.get(attack_type, []):
            if keyword in payload_lower:
                blocked = True
                break
        
        # 添加随机性
        import random
        if random.random() < self.block_rate:
            blocked = True
        
        if blocked:
            self.blocked_count += 1
            status_code = 403
            response_text = "403 Forbidden - Blocked by WAF"
        else:
            status_code = 200
            response_text = "200 OK - Request passed"
        
        return WAFResponse(
            blocked=blocked,
            status_code=status_code,
            response_text=response_text,
            response_time=0.01
        )
    
    def test_connection(self) -> bool:
        """模拟环境总是可用"""
        return True


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("WAF环境测试")
    print("="*60)
    
    # 使用模拟环境
    waf = MockWAFEnvironment(block_rate=0.7)
    
    # 测试payload
    test_payloads = [
        "' OR 1=1 --",
        "UNION SELECT * FROM users",
        "<script>alert(1)</script>",
        "normal input text"
    ]
    
    print("\n测试payload:")
    for payload in test_payloads:
        response = waf.send_payload(payload, attack_type="sqli")
        status = "🔴 拦截" if response.blocked else "✅ 绕过"
        print(f"{status} | {payload[:50]}")
    
    # 统计
    print("\n" + "="*60)
    print("统计信息:")
    stats = waf.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
