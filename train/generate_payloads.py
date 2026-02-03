"""
使用训练好的RL模型批量生成WAF绕过载荷
支持多种生成策略和输出格式
"""

import os
import sys
import json
import torch
import argparse
from typing import List, Dict, Set
from datetime import datetime
from tqdm import tqdm
from transformers import GPT2Tokenizer, AutoModelForCausalLM


class PayloadGenerator:
    """载荷生成器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化生成器
        
        Args:
            model_path: RL模型路径
            device: 运行设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"📦 加载模型: {model_path}")
        
        # 加载模型
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型已加载 (设备: {self.device})")
    
    def generate_batch(
        self,
        batch_size: int = 10,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> List[str]:
        """
        批量生成载荷
        
        Args:
            batch_size: 批次大小
            max_length: 最大长度
            temperature: 温度参数
            top_k: Top-K采样
            top_p: Nucleus采样
            do_sample: 是否采样
            num_beams: Beam search束宽
            
        Returns:
            生成的载荷列表
        """
        payloads = []
        
        with torch.no_grad():
            # 创建批次输入
            input_ids = torch.tensor(
                [[self.tokenizer.bos_token_id]] * batch_size,
                dtype=torch.long,
                device=self.device
            )
            
            attention_mask = torch.ones_like(input_ids)

            # 生成
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
            
            # 解码
            for output in outputs:
                payload = self.tokenizer.decode(output, skip_special_tokens=True)
                payloads.append(payload)
        
        return payloads
    
    def generate_diverse(
        self,
        num_samples: int = 100,
        batch_size: int = 10,
        max_length: int = 128,
        temperature_range: tuple = (0.8, 1.2),
        show_progress: bool = True,
    ) -> List[str]:
        """
        生成多样化的载荷（使用不同温度）
        
        Args:
            num_samples: 总样本数
            batch_size: 批次大小
            max_length: 最大长度
            temperature_range: 温度范围
            show_progress: 显示进度条
            
        Returns:
            生成的载荷列表
        """
        payloads = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        iterator = tqdm(range(num_batches), desc="生成载荷") if show_progress else range(num_batches)
        
        for i in iterator:
            # 动态温度
            temp = temperature_range[0] + (temperature_range[1] - temperature_range[0]) * (i / num_batches)
            
            # 当前批次大小
            current_batch_size = min(batch_size, num_samples - len(payloads))
            
            # 生成
            batch = self.generate_batch(
                batch_size=current_batch_size,
                max_length=max_length,
                temperature=temp,
            )
            
            payloads.extend(batch)
        
        return payloads[:num_samples]
    
    def deduplicate(self, payloads: List[str]) -> List[str]:
        """去重"""
        return list(dict.fromkeys(payloads))  # 保持顺序的去重
    
    def filter_valid(self, payloads: List[str], min_length: int = 5) -> List[str]:
        """过滤无效载荷"""
        return [p for p in payloads if len(p.strip()) >= min_length]
    
    def save_to_file(
        self,
        payloads: List[str],
        output_file: str,
        format: str = "txt"
    ):
        """
        保存到文件
        
        Args:
            payloads: 载荷列表
            output_file: 输出文件
            format: 格式 (txt/json/csv)
        """
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if format == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                for payload in payloads:
                    f.write(f"{payload}\n")
        
        elif format == "json":
            data = {
                'generated_at': datetime.now().isoformat(),
                'total': len(payloads),
                'payloads': payloads
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['payload'])
                for payload in payloads:
                    writer.writerow([payload])
        
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"✓ 已保存 {len(payloads)} 个载荷到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="批量生成WAF绕过载荷")
    
    # 必需参数
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="RL模型路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出文件路径"
    )
    
    # 生成参数
    parser.add_argument("--num_samples", type=int, default=1000, help="生成数量")
    parser.add_argument("--batch_size", type=int, default=10, help="批次大小")
    parser.add_argument("--max_length", type=int, default=128, help="最大长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度 (固定值)")
    parser.add_argument("--use_diverse", action="store_true", help="使用多样化生成（动态温度）")
    parser.add_argument("--temp_min", type=float, default=0.8, help="最小温度")
    parser.add_argument("--temp_max", type=float, default=1.2, help="最大温度")
    
    # 后处理
    parser.add_argument("--deduplicate", action="store_true", help="去重")
    parser.add_argument("--min_length", type=int, default=5, help="最小长度")
    
    # 输出格式
    parser.add_argument("--format", type=str, default="txt", choices=["txt", "json", "csv"], help="输出格式")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # 检查模型
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型不存在: {args.model_path}")
        sys.exit(1)
    
    print("="*80)
    print("🚀 载荷批量生成工具")
    print("="*80)
    print(f"模型: {args.model_path}")
    print(f"生成数量: {args.num_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"输出文件: {args.output_file}")
    print(f"格式: {args.format}")
    print("="*80 + "\n")
    
    # 创建生成器
    generator = PayloadGenerator(args.model_path, args.device)
    
    # 生成载荷
    print(f"\n📝 开始生成 {args.num_samples} 个载荷...")
    start_time = datetime.now()
    
    if args.use_diverse:
        print(f"   使用多样化生成 (温度: {args.temp_min} ~ {args.temp_max})")
        payloads = generator.generate_diverse(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            temperature_range=(args.temp_min, args.temp_max),
            show_progress=True,
        )
    else:
        print(f"   使用固定温度: {args.temperature}")
        payloads = []
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        
        for i in tqdm(range(num_batches), desc="生成载荷"):
            current_batch_size = min(args.batch_size, args.num_samples - len(payloads))
            batch = generator.generate_batch(
                batch_size=current_batch_size,
                max_length=args.max_length,
                temperature=args.temperature,
            )
            payloads.extend(batch)
        
        payloads = payloads[:args.num_samples]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✓ 生成完成! 耗时: {duration:.2f}秒")
    print(f"  生成速度: {len(payloads)/duration:.2f} 个/秒")
    
    # 后处理
    original_count = len(payloads)
    
    # 去重
    if args.deduplicate:
        print(f"\n🔄 去重...")
        payloads = generator.deduplicate(payloads)
        print(f"  去重前: {original_count}")
        print(f"  去重后: {len(payloads)}")
        print(f"  重复率: {(1 - len(payloads)/original_count)*100:.1f}%")
    
    # 过滤
    payloads = generator.filter_valid(payloads, args.min_length)
    print(f"\n📊 最终统计:")
    print(f"  有效载荷: {len(payloads)}")
    print(f"  平均长度: {sum(len(p) for p in payloads)/len(payloads):.1f} 字符")
    
    # 保存
    print(f"\n💾 保存到文件...")
    generator.save_to_file(payloads, args.output_file, args.format)
    
    # 显示样例
    print(f"\n📄 样例 (前5个):")
    print("="*80)
    for i, payload in enumerate(payloads[:5]):
        print(f"\n[{i+1}] {payload}")
    print("="*80)
    
    print(f"\n✅ 完成!")
    print(f"   输出文件: {args.output_file}")
    print(f"   载荷数量: {len(payloads)}")


if __name__ == "__main__":
    main()
