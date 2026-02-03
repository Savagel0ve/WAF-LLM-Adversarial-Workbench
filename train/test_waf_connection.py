"""
测试 WAF 连接和基本功能

用于验证 WAF 服务是否正常运行并能正确响应
"""
import argparse
import logging
from waf_env import WAFEnvironment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_waf_connection(waf_url: str, attack_type: str = "sqli"):
    """测试 WAF 连接"""
    
    logger.info("="*60)
    logger.info("WAF 连接测试")
    logger.info("="*60)
    logger.info(f"WAF URL: {waf_url}")
    logger.info(f"攻击类型: {attack_type}")
    
    # 初始化 WAF 环境
    waf_env = WAFEnvironment(
        waf_type="modsecurity",
        modsecurity_url=waf_url,
        timeout=10,
        max_retries=3
    )
    
    # 测试连接
    logger.info("\n测试基本连接...")
    if not waf_env.test_connection():
        logger.error("❌ WAF 连接失败!")
        logger.error("请检查:")
        logger.error("  1. WAF 服务是否正在运行")
        logger.error("  2. URL 是否正确")
        logger.error("  3. 防火墙设置")
        return False
    
    logger.info("✅ WAF 连接成功!")
    
    # 测试payload
    logger.info("\n" + "="*60)
    logger.info("测试不同类型的 Payload")
    logger.info("="*60)
    
    test_payloads = {
        "sqli": [
            ("正常输入", "hello world", False),
            ("基础注入", "' OR 1=1 --", True),
            ("UNION注入", "UNION SELECT * FROM users", True),
            ("堆叠查询", "1'; DROP TABLE users--", True),
        ],
        "xss": [
            ("正常输入", "hello world", False),
            ("script标签", "<script>alert(1)</script>", True),
            ("img标签", "<img src=x onerror=alert(1)>", True),
            ("事件处理", "javascript:alert(1)", True),
        ],
        "rce": [
            ("正常输入", "hello world", False),
            ("命令注入", "; ls -la", True),
            ("管道命令", "| cat /etc/passwd", True),
            ("命令替换", "$(whoami)", True),
        ]
    }
    
    payloads = test_payloads.get(attack_type, test_payloads["sqli"])
    
    results = []
    
    for name, payload, should_block in payloads:
        logger.info(f"\n测试: {name}")
        logger.info(f"Payload: {payload}")
        
        response = waf_env.send_payload(payload, attack_type)
        
        # 判断结果
        blocked_str = "拦截" if response.blocked else "通过"
        status_emoji = "🔴" if response.blocked else "🟢"
        
        logger.info(f"结果: {status_emoji} {blocked_str} (状态码: {response.status_code})")
        logger.info(f"响应时间: {response.response_time:.3f}s")
        logger.info(f"响应内容: {response.response_text[:200]}")
        
        # 验证是否符合预期
        if should_block and not response.blocked:
            logger.warning("⚠️  警告: 应该被拦截但未拦截!")
            results.append((name, False))
        elif not should_block and response.blocked:
            logger.warning("⚠️  警告: 不应被拦截但被拦截!")
            results.append((name, False))
        else:
            logger.info("✅ 符合预期")
            results.append((name, True))
        
        if response.error:
            logger.error(f"错误: {response.error}")
    
    # 统计结果
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    
    stats = waf_env.get_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # 准确性
    correct = sum(1 for _, result in results if result)
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    logger.info(f"\n测试准确性: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        logger.info("✅ WAF 工作正常!")
        return True
    else:
        logger.warning("⚠️  WAF 可能存在配置问题")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 WAF 连接")
    parser.add_argument(
        "--waf_url",
        type=str,
        default="http://localhost:8081",
        help="WAF URL"
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="sqli",
        choices=["sqli", "xss", "rce"],
        help="攻击类型"
    )
    
    args = parser.parse_args()
    
    success = test_waf_connection(args.waf_url, args.attack_type)
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("✅ 所有测试通过!")
        logger.info("="*60)
        logger.info("\n可以开始训练奖励模型了。")
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ 测试失败!")
        logger.error("="*60)
        logger.error("\n请检查 WAF 配置后重试。")


if __name__ == "__main__":
    main()
