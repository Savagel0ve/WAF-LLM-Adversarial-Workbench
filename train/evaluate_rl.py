"""
评估RL模型在真实WAF上的绕过率
支持ModSecurity和Naxsi WAF
"""

import os
import sys
import json
import random
import re
import torch
import argparse
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from transformers import GPT2Tokenizer, AutoModelForCausalLM

try:
    from waf_env import ModSecurityWAF, NaxsiWAF
except ImportError:
    print("警告: 无法导入waf_env，请确保waf_env.py存在")
    ModSecurityWAF = None
    NaxsiWAF = None

try:
    from verifier import SQLiVerifier
except ImportError:
    SQLiVerifier = None


class RLEvaluator:
    """RL模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        waf_url: str = "http://localhost:8001",
        waf_type: str = "modsecurity",
        device: str = "cuda"
    ):
        """
        初始化评估器
        
        Args:
            model_path: RL模型路径
            waf_url: WAF URL
            waf_type: WAF类型 (modsecurity/naxsi)
            device: 运行设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.waf_url = waf_url
        self.waf_type = waf_type
        
        # 加载模型
        print(f"📦 加载模型: {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化WAF
        print(f"🔒 连接WAF: {waf_url} ({waf_type})")
        if waf_type == "modsecurity" and ModSecurityWAF:
            self.waf = ModSecurityWAF(waf_url)
        elif waf_type == "naxsi" and NaxsiWAF:
            self.waf = NaxsiWAF(waf_url)
        else:
            print(f"警告: 无法初始化WAF，将使用HTTP请求模拟")
            self.waf = None
        
        # 测试WAF连接
        if not self._test_waf_connection():
            print(f"⚠️  警告: 无法连接到WAF {waf_url}")
            print(f"   请确保WAF服务正在运行")
        else:
            print(f"✓ WAF连接成功")
        
        # 初始化验证器
        self.verifier = SQLiVerifier() if SQLiVerifier else None
        
        print(f"✓ 评估器初始化完成")
    
    def _test_waf_connection(self) -> bool:
        """测试WAF连接"""
        try:
            if self.waf:
                # 使用WAF对象测试
                result = self.waf.test("' OR 1=1--")
                return True
            else:
                # 使用HTTP请求测试
                response = requests.get(self.waf_url, timeout=5)
                return response.status_code in [200, 403]
        except Exception as e:
            return False

    def _extract_dvwa_token(self, html: str) -> Optional[str]:
        match = re.search(r"name=['\"]user_token['\"]\s+value=['\"]([^'\"]+)['\"]", html)
        return match.group(1) if match else None

    def _dvwa_login(
        self,
        username: str,
        password: str,
        login_url: str,
        security_url: str,
        security_level: str,
    ) -> Optional[requests.Session]:
        """登录DVWA并设置安全级别，返回已登录的Session"""
        try:
            session = requests.Session()
            login_page = session.get(login_url, timeout=10)
            token = self._extract_dvwa_token(login_page.text)

            payload = {
                "username": username,
                "password": password,
                "Login": "Login",
            }
            if token:
                payload["user_token"] = token

            login_resp = session.post(login_url, data=payload, timeout=10)
            if login_resp.status_code != 200:
                return None

            # 设置安全级别
            sec_page = session.get(security_url, timeout=10)
            sec_token = self._extract_dvwa_token(sec_page.text)
            sec_payload = {"security": security_level, "seclev_submit": "Submit"}
            if sec_token:
                sec_payload["user_token"] = sec_token
            session.post(security_url, data=sec_payload, timeout=10)

            return session
        except Exception:
            return None
    
    def generate_payloads(
        self,
        num_samples: int = 100,
        max_length: int = 128,
        temperature: float = 1.0,
        batch_size: int = 10,
        start_prompt: str = "<start>",
    ) -> List[str]:
        """
        生成载荷
        
        Args:
            num_samples: 样本数量
            max_length: 最大长度
            temperature: 温度
            batch_size: 批次大小
            
        Returns:
            生成的载荷列表
        """
        print(f"\n🎲 生成 {num_samples} 个载荷...")
        
        payloads = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="生成"):
                current_batch_size = min(batch_size, num_samples - len(payloads))
                
                # 创建输入
                prompt_ids = self.tokenizer.encode(start_prompt, add_special_tokens=False)
                if not prompt_ids:
                    prompt_ids = [self.tokenizer.eos_token_id]
                prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
                input_ids = prompt_tensor.unsqueeze(0).repeat(current_batch_size, 1)
                attention_mask = torch.ones_like(input_ids)
                
                # 生成
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # 解码
                for output in outputs:
                    payload = self.tokenizer.decode(output, skip_special_tokens=True)
                    payloads.append(payload)
        
        return payloads[:num_samples]
    
    def test_payload_on_waf(self, payload: str) -> Tuple[bool, str]:
        """
        在WAF上测试单个载荷
        
        Args:
            payload: 测试载荷
            
        Returns:
            (是否绕过, 响应信息)
        """
        try:
            if self.waf:
                result = self.waf.test(payload)
                if isinstance(result, dict):
                    blocked = result.get('blocked', True)
                    return not blocked, result.get('message', '')
                else:
                    # 简单的布尔结果
                    return result, ''
            else:
                # 使用HTTP请求模拟
                response = requests.get(
                    self.waf_url,
                    params={'q': payload},
                    timeout=5
                )
                # 403表示被阻止，200表示通过
                bypassed = response.status_code == 200
                return bypassed, f"HTTP {response.status_code}"
        
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def _functional_verify(
        self,
        payloads: List[str],
        url: Optional[str],
        param_name: str,
        method: str,
        success_regex: Optional[str],
        sample_size: int,
        headers: Optional[Dict[str, str]],
        cookies: Optional[Dict[str, str]],
        timeout: int,
        session: Optional[requests.Session] = None,
    ) -> Dict:
        """功能性验证：验证payload是否能在后端成功执行"""
        if not payloads:
            return {
                "functional_checked": 0,
                "functional_success": 0,
                "functional_rate": None,
                "manual_sample_file": None,
            }

        samples = payloads[:]
        random.shuffle(samples)
        samples = samples[:sample_size]

        if not url:
            # 输出人工检查样本
            sample_file = os.path.join("results", "functional_verification_samples.json")
            os.makedirs(os.path.dirname(sample_file), exist_ok=True)
            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "count": len(samples),
                        "samples": samples,
                        "note": "No functional verification URL provided. Please manually verify these samples.",
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return {
                "functional_checked": 0,
                "functional_success": 0,
                "functional_rate": None,
                "manual_sample_file": sample_file,
            }

        success_count = 0
        checked = 0
        regex = re.compile(success_regex, re.IGNORECASE) if success_regex else None

        for payload in tqdm(samples, desc="功能验证"):
            try:
                client = session if session else requests
                if method.lower() == "post":
                    response = client.post(
                        url,
                        data={param_name: payload},
                        headers=headers,
                        cookies=cookies,
                        timeout=timeout,
                    )
                else:
                    response = client.get(
                        url,
                        params={param_name: payload},
                        headers=headers,
                        cookies=cookies,
                        timeout=timeout,
                    )
                checked += 1

                if regex:
                    if regex.search(response.text):
                        success_count += 1
                else:
                    # 如果没有提供regex，默认使用200作为成功
                    if response.status_code == 200:
                        success_count += 1
            except Exception:
                continue

        functional_rate = (success_count / checked * 100) if checked > 0 else None
        return {
            "functional_checked": checked,
            "functional_success": success_count,
            "functional_rate": functional_rate,
            "manual_sample_file": None,
        }

    def evaluate(
        self,
        num_samples: int = 100,
        max_length: int = 128,
        temperature: float = 1.0,
        batch_size: int = 10,
        start_prompt: str = "<start>",
        functional_verify: bool = False,
        fv_url: Optional[str] = None,
        fv_param: str = "id",
        fv_method: str = "get",
        fv_success_regex: Optional[str] = None,
        fv_sample_size: int = 100,
        fv_headers: Optional[Dict[str, str]] = None,
        fv_cookies: Optional[Dict[str, str]] = None,
        fv_timeout: int = 10,
        fv_session: Optional[requests.Session] = None,
    ) -> Dict:
        """
        完整评估流程
        
        Args:
            num_samples: 测试样本数
            max_length: 最大长度
            temperature: 温度
            batch_size: 生成批次大小
            
        Returns:
            评估结果字典
        """
        print("\n" + "="*80)
        print("📊 开始评估")
        print("="*80)
        
        # Step 1: 生成载荷
        payloads = self.generate_payloads(
            num_samples=num_samples,
            max_length=max_length,
            temperature=temperature,
            batch_size=batch_size,
            start_prompt=start_prompt,
        )
        
        # Step 2: 去重
        original_count = len(payloads)
        payloads = list(dict.fromkeys(payloads))
        print(f"\n去重: {original_count} -> {len(payloads)}")
        
        # Step 3: 语法验证
        valid_payloads = []
        if self.verifier:
            print(f"\n✓ 验证语法...")
            for payload in tqdm(payloads, desc="验证"):
                if self.verifier.verify(payload):
                    valid_payloads.append(payload)
            print(f"  语法有效: {len(valid_payloads)}/{len(payloads)} ({len(valid_payloads)/len(payloads)*100:.1f}%)")
        else:
            print(f"\n⚠️  跳过语法验证")
            valid_payloads = payloads
        
        # Step 4: WAF测试
        print(f"\n🔒 测试WAF绕过...")
        bypassed_payloads = []
        blocked_payloads = []
        error_payloads = []
        
        for payload in tqdm(valid_payloads, desc="测试WAF"):
            bypassed, message = self.test_payload_on_waf(payload)
            
            if "Error" in message or "Timeout" in message:
                error_payloads.append((payload, message))
            elif bypassed:
                bypassed_payloads.append(payload)
            else:
                blocked_payloads.append(payload)
        
        # 计算指标
        total = len(valid_payloads)
        num_bypassed = len(bypassed_payloads)
        num_blocked = len(blocked_payloads)
        num_errors = len(error_payloads)
        
        bypass_rate = (num_bypassed / total * 100) if total > 0 else 0
        er = (num_bypassed / original_count * 100) if original_count > 0 else 0
        nrr = (len(payloads) / original_count * 100) if original_count > 0 else 0
        
        # 结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model.__class__.__name__,
            'waf_url': self.waf_url,
            'waf_type': self.waf_type,
            'total_generated': original_count,
            'unique_payloads': len(payloads),
            'valid_payloads': len(valid_payloads),
            'tested_payloads': total,
            'tp': num_bypassed,
            'er': er,
            'nrr': nrr,
            'bypassed': num_bypassed,
            'blocked': num_blocked,
            'errors': num_errors,
            'bypass_rate': bypass_rate,
            'valid_rate': (len(valid_payloads) / len(payloads) * 100) if payloads else 0,
            'bypassed_samples': bypassed_payloads[:10],  # 保存前10个成功样例
        }
        
        # 打印结果
        print("\n" + "="*80)
        print("📊 评估结果")
        print("="*80)
        print(f"\n生成统计:")
        print(f"  - 总生成: {original_count}")
        print(f"  - 唯一: {len(payloads)} ({len(payloads)/original_count*100:.1f}%)")
        print(f"  - 语法有效: {len(valid_payloads)} ({results['valid_rate']:.1f}%)")
        
        print(f"\nWAF测试:")
        print(f"  - 测试总数: {total}")
        print(f"  - 绕过: {num_bypassed} ({bypass_rate:.1f}%)")
        print(f"  - 被阻止: {num_blocked} ({num_blocked/total*100 if total > 0 else 0:.1f}%)")
        print(f"  - 错误: {num_errors}")
        
        print(f"\n指标:")
        print(f"  - TP: {num_bypassed}")
        print(f"  - ER: {er:.2f}%")
        print(f"  - NRR: {nrr:.2f}%")

        # 功能性验证
        if functional_verify:
            print(f"\n🧪 功能性验证...")
            fv_result = self._functional_verify(
                bypassed_payloads,
                fv_url,
                fv_param,
                fv_method,
                fv_success_regex,
                fv_sample_size,
                fv_headers,
                fv_cookies,
                fv_timeout,
            fv_session,
            )
            results.update(fv_result)
            if fv_result.get("manual_sample_file"):
                print(f"  - 已输出人工验证样本: {fv_result['manual_sample_file']}")
            else:
                print(f"  - 功能性验证成功: {fv_result['functional_success']}/{fv_result['functional_checked']}")
                if fv_result["functional_rate"] is not None:
                    print(f"  - 成功率: {fv_result['functional_rate']:.2f}%")

        print(f"\n🎯 最终绕过率: {bypass_rate:.2f}%")
        
        if bypassed_payloads:
            print(f"\n✓ 成功绕过的样例 (前5个):")
            print("-"*80)
            for i, payload in enumerate(bypassed_payloads[:5]):
                print(f"[{i+1}] {payload}")
            print("-"*80)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="评估RL模型WAF绕过率")
    
    # 必需参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="RL模型路径"
    )
    
    # WAF配置
    parser.add_argument(
        "--waf_url",
        type=str,
        default="http://localhost:8001",
        help="WAF URL"
    )
    parser.add_argument(
        "--waf_type",
        type=str,
        default="modsecurity",
        choices=["modsecurity", "naxsi"],
        help="WAF类型"
    )
    
    # 评估参数
    parser.add_argument("--num_samples", type=int, default=100, help="测试样本数")
    parser.add_argument("--max_length", type=int, default=128, help="最大长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度")
    parser.add_argument("--batch_size", type=int, default=10, help="生成批次大小")
    parser.add_argument("--start_prompt", type=str, default="<start>", help="生成起始prompt")
    
    # 输出
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="保存结果的JSON文件"
    )

    # 功能性验证参数
    parser.add_argument("--functional_verify", action="store_true", help="启用功能性验证")
    parser.add_argument("--fv_url", type=str, default=None, help="后端应用URL (如DVWA SQLi页面)")
    parser.add_argument("--fv_param", type=str, default="id", help="注入参数名")
    parser.add_argument("--fv_method", type=str, default="get", choices=["get", "post"], help="请求方法")
    parser.add_argument("--fv_success_regex", type=str, default=None, help="成功判定正则")
    parser.add_argument("--fv_sample_size", type=int, default=100, help="验证样本数")
    parser.add_argument("--fv_headers", type=str, default=None, help="JSON格式请求头")
    parser.add_argument("--fv_cookies", type=str, default=None, help="JSON格式cookies")
    parser.add_argument("--fv_timeout", type=int, default=10, help="请求超时(秒)")
    parser.add_argument("--dvwa_login", action="store_true", help="自动登录DVWA并进行功能验证")
    parser.add_argument("--dvwa_username", type=str, default="admin", help="DVWA用户名")
    parser.add_argument("--dvwa_password", type=str, default="password", help="DVWA密码")
    parser.add_argument("--dvwa_login_url", type=str, default="http://localhost:8081/login.php", help="DVWA登录URL")
    parser.add_argument("--dvwa_security_url", type=str, default="http://localhost:8081/security.php", help="DVWA安全级别URL")
    parser.add_argument("--dvwa_security_level", type=str, default="low", help="DVWA安全级别(low/medium/high)")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 检查模型
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型不存在: {args.model_path}")
        sys.exit(1)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("="*80)
    print("🧪 RL模型WAF绕过评估")
    print("="*80)
    print(f"模型: {args.model_path}")
    print(f"WAF: {args.waf_url} ({args.waf_type})")
    print(f"测试样本: {args.num_samples}")
    print("="*80)
    
    # 解析headers/cookies
    headers = json.loads(args.fv_headers) if args.fv_headers else None
    cookies = json.loads(args.fv_cookies) if args.fv_cookies else None

    # 创建评估器
    evaluator = RLEvaluator(
        model_path=args.model_path,
        waf_url=args.waf_url,
        waf_type=args.waf_type,
        device=args.device,
    )
    
    # 开始评估
    try:
        dvwa_session = None
        if args.dvwa_login:
            dvwa_session = evaluator._dvwa_login(
                args.dvwa_username,
                args.dvwa_password,
                args.dvwa_login_url,
                args.dvwa_security_url,
                args.dvwa_security_level,
            )
            if not dvwa_session:
                print("⚠️  DVWA登录失败，将跳过自动功能验证。")
                args.functional_verify = False

        results = evaluator.evaluate(
            num_samples=args.num_samples,
            max_length=args.max_length,
            temperature=args.temperature,
            batch_size=args.batch_size,
            start_prompt=args.start_prompt,
            functional_verify=args.functional_verify,
            fv_url=args.fv_url,
            fv_param=args.fv_param,
            fv_method=args.fv_method,
            fv_success_regex=args.fv_success_regex,
            fv_sample_size=args.fv_sample_size,
            fv_headers=headers,
            fv_cookies=cookies,
            fv_timeout=args.fv_timeout,
            fv_session=dvwa_session,
        )
        
        # 保存结果
        if args.output_file:
            print(f"\n💾 保存结果到: {args.output_file}")
            os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ 结果已保存")
        
        print("\n" + "="*80)
        print("✅ 评估完成!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
