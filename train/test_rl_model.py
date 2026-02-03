"""
测试强化学习训练后的模型
支持 Qwen2.5-Coder 和其他现代 LLM

生成 SQL 注入载荷并评估质量
"""

import os
import sys
import torch
import argparse
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from verifier import SQLiVerifier
except ImportError:
    SQLiVerifier = None
    print("警告: 无法导入SQLiVerifier，将跳过语法验证")


class RLModelTester:
    """RL模型测试器 - 支持 Qwen2.5-Coder 和其他现代 LLM"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化测试器
        
        Args:
            model_path: RL训练后的模型路径
            device: 运行设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"🔍 加载模型: {model_path}")
        
        # 加载 tokenizer (使用 AutoTokenizer 支持各种模型)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device != "cuda":
            self.model.to(self.device)
        
        self.model.eval()
        
        # 初始化验证器
        self.verifier = SQLiVerifier() if SQLiVerifier else None
        
        print(f"   模型类型: {type(self.model).__name__}")
        print(f"✅ 模型加载完成 (设备: {self.device})")
    
    def generate_payloads(
        self,
        num_samples: int = 10,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[str]:
        """
        生成载荷
        
        Args:
            num_samples: 生成数量
            max_length: 最大长度
            temperature: 温度参数 (越高越随机)
            top_k: Top-K采样
            top_p: Nucleus采样
            do_sample: 是否使用采样
            
        Returns:
            生成的载荷列表
        """
        print(f"\n🎲 生成 {num_samples} 个载荷...")
        print(f"   参数: temp={temperature}, top_k={top_k}, top_p={top_p}")
        
        payloads = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # 创建起始输入 (BOS token)
                input_ids = torch.tensor(
                    [[self.tokenizer.bos_token_id]],
                    dtype=torch.long,
                    device=self.device
                )
                
                attention_mask = torch.ones_like(input_ids)

                # 生成
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
                
                # 解码
                payload = self.tokenizer.decode(output[0], skip_special_tokens=True)
                payloads.append(payload)
                
                # 实时显示
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"   [{i+1}/{num_samples}] {payload[:80]}...")
        
        print(f"✓ 生成完成")
        return payloads
    
    def evaluate_payloads(self, payloads: List[str]) -> Dict:
        """
        评估生成的载荷
        
        Args:
            payloads: 载荷列表
            
        Returns:
            评估结果字典
        """
        print(f"\n📊 评估 {len(payloads)} 个载荷...")
        
        results = {
            'total': len(payloads),
            'valid': 0,
            'invalid': 0,
            'avg_length': 0,
            'unique': 0,
            'duplicates': 0,
        }
        
        # 去重
        unique_payloads = list(set(payloads))
        results['unique'] = len(unique_payloads)
        results['duplicates'] = len(payloads) - len(unique_payloads)
        
        # 平均长度
        lengths = [len(p) for p in payloads]
        results['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
        
        # 语法验证
        if self.verifier:
            print("   正在验证SQL语法...")
            for payload in payloads:
                if self.verifier.verify(payload):
                    results['valid'] += 1
                else:
                    results['invalid'] += 1
            
            results['valid_rate'] = results['valid'] / results['total'] * 100
        else:
            print("   (跳过语法验证)")
            results['valid'] = -1
            results['invalid'] = -1
            results['valid_rate'] = -1
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"  - 总数: {results['total']}")
        print(f"  - 唯一: {results['unique']} ({results['unique']/results['total']*100:.1f}%)")
        print(f"  - 重复: {results['duplicates']}")
        print(f"  - 平均长度: {results['avg_length']:.1f} 字符")
        
        if results['valid'] >= 0:
            print(f"  - 有效: {results['valid']} ({results['valid_rate']:.1f}%)")
            print(f"  - 无效: {results['invalid']}")
        
        return results
    
    def show_samples(self, payloads: List[str], num_show: int = 10):
        """显示样例载荷"""
        print(f"\n📄 载荷样例 (前{num_show}个):")
        print("="*80)
        
        for i, payload in enumerate(payloads[:num_show]):
            print(f"\n[{i+1}]")
            print(payload)
            print("-"*80)


def main():
    parser = argparse.ArgumentParser(description="测试RL训练后的模型")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="RL模型路径 (例如: ./models/rl_sqli_gpt2/final_model)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="生成样本数量"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="生成温度 (0.1-2.0, 越高越随机)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K采样"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus采样"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="保存生成结果的文件路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型不存在: {args.model_path}")
        sys.exit(1)
    
    print("="*80)
    print("🧪 RL模型测试工具")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"生成数量: {args.num_samples}")
    print(f"温度: {args.temperature}")
    print("="*80)
    
    # 创建测试器
    tester = RLModelTester(args.model_path, args.device)
    
    # 生成载荷
    payloads = tester.generate_payloads(
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # 评估
    results = tester.evaluate_payloads(payloads)
    
    # 显示样例
    tester.show_samples(payloads, num_show=min(10, len(payloads)))
    
    # 保存结果
    if args.output_file:
        print(f"\n💾 保存结果到: {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, payload in enumerate(payloads):
                f.write(f"{payload}\n")
        print(f"✓ 已保存 {len(payloads)} 个载荷")
    
    print("\n" + "="*80)
    print("✓ 测试完成!")
    print("="*80)
    
    # 返回评估结果
    return results


if __name__ == "__main__":
    main()
