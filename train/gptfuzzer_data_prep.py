"""
GPTFuzzer数据预处理脚本
按照论文方法进行tokenization和格式化
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from transformers import GPT2Tokenizer
from datasets import Dataset, DatasetDict
from tqdm import tqdm


class GPTFuzzerDataPreprocessor:
    """
    GPTFuzzer数据预处理器
    
    按照论文方法:
    1. 加载原始payload数据
    2. 添加攻击类型标记
    3. Tokenization
    4. 保存为HuggingFace datasets格式
    """
    
    def __init__(self, 
                 data_dir="data/processed",
                 output_dir="data/gptfuzzer",
                 model_name="gpt2"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据GPTFuzzer论文的数据量要求
        self.max_samples = {
            "sqli": 512000,  # 512K
            "xss": 512000,   # 512K
            "rce": None      # RCE使用全量数据 (37,302)
        }
        
        # 加载tokenizer
        print(f"加载tokenizer: {model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 添加攻击类型特殊token
        special_tokens = {
            "additional_special_tokens": ["<SQLI>", "<XSS>", "<RCE>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        print(f"词表大小: {len(self.tokenizer)}")
        print(f"特殊token: {self.tokenizer.special_tokens_map}")
    
    def load_attack_data(self, attack_type: str) -> Dict[str, List[str]]:
        """
        加载指定攻击类型的数据
        
        Returns:
            {"train": [...], "val": [...], "test": [...]}
        """
        attack_dir = self.data_dir / attack_type
        if not attack_dir.exists():
            print(f"警告: {attack_dir} 不存在")
            return {}
        
        splits = {}
        for split in ["train", "val", "test"]:
            txt_file = attack_dir / f"{split}.txt"
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    payloads = [line.strip() for line in f if line.strip()]
                splits[split] = payloads
                print(f"  加载 {attack_type}/{split}: {len(payloads)} 条")
        
        return splits
    
    def format_payload(self, payload: str, attack_type: str) -> str:
        """
        格式化payload,添加攻击类型前缀
        
        按照GPTFuzzer论文格式: <TYPE>payload
        """
        type_token = f"<{attack_type.upper()}>"
        return f"{type_token} {payload}"
    
    def create_dataset(self, attack_type: str, max_length: int = 128) -> DatasetDict:
        """
        创建指定攻击类型的HuggingFace Dataset
        
        Args:
            attack_type: 攻击类型 ("sqli", "xss", "rce")
            max_length: 最大序列长度
            
        Returns:
            DatasetDict with train/val/test splits
        """
        print(f"\n{'='*60}")
        print(f"处理 {attack_type.upper()} 数据集")
        max_size = self.max_samples.get(attack_type.lower())
        if max_size:
            print(f"论文要求数据量: {max_size:,} 条")
        else:
            print(f"论文要求: 使用全量数据")
        print(f"{'='*60}")
        
        # 加载数据
        splits = self.load_attack_data(attack_type)
        if not splits:
            return None
        
        # 验证数据量是否符合论文要求
        total_loaded = sum(len(payloads) for payloads in splits.values())
        if max_size and total_loaded > max_size * 1.1:  # 允许10%误差
            print(f"⚠️  警告: 加载的数据量 ({total_loaded:,}) 远超论文要求 ({max_size:,})")
            print(f"    请先运行 prepare_data.py 重新处理数据集")
        
        dataset_dict = {}
        
        for split_name, payloads in splits.items():
            print(f"\n处理 {split_name} 集...")
            
            # 格式化payload
            formatted_texts = []
            for payload in tqdm(payloads, desc=f"格式化{split_name}"):
                formatted = self.format_payload(payload, attack_type)
                formatted_texts.append(formatted)
            
            # Tokenization
            print(f"Tokenizing {split_name}...")
            encodings = self.tokenizer(
                formatted_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None  # 返回列表而不是tensor
            )
            
            # 创建Dataset
            dataset = Dataset.from_dict({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "text": formatted_texts  # 保留原始文本用于调试
            })
            
            dataset_dict[split_name] = dataset
            
            print(f"  {split_name}: {len(dataset)} 样本")
            print(f"  平均token数: {sum(len(ids) for ids in encodings['input_ids']) / len(encodings['input_ids']):.1f}")
        
        # 创建DatasetDict
        dataset_dict_obj = DatasetDict(dataset_dict)
        
        # 保存
        output_path = self.output_dir / attack_type
        dataset_dict_obj.save_to_disk(str(output_path))
        print(f"\n✅ 保存到: {output_path}")
        
        # 保存tokenizer配置
        tokenizer_path = output_path / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))
        print(f"✅ Tokenizer保存到: {tokenizer_path}")
        
        # 保存元数据
        metadata = {
            "attack_type": attack_type,
            "model_name": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "max_length": max_length,
            "splits": {
                split: len(dataset_dict[split])
                for split in dataset_dict.keys()
            }
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return dataset_dict_obj
    
    def create_all_datasets(self, max_length: int = 128):
        """处理所有攻击类型的数据"""
        print("\n" + "="*60)
        print("GPTFuzzer数据预处理")
        print("="*60)
        print("论文要求的数据量:")
        print("  - SQLi: 512,000 条")
        print("  - XSS:  512,000 条")
        print("  - RCE:  37,302 条 (全量)")
        print("="*60)
        
        attack_types = ["xss", "rce", "sqli"]
        results = {}
        
        for attack_type in attack_types:
            try:
                dataset = self.create_dataset(attack_type, max_length)
                if dataset:
                    results[attack_type] = dataset
            except Exception as e:
                print(f"❌ 处理 {attack_type} 失败: {e}")
        
        # 生成总结报告
        print("\n" + "="*60)
        print("处理完成总结")
        print("="*60)
        for attack_type, dataset in results.items():
            print(f"\n{attack_type.upper()}:")
            for split, data in dataset.items():
                print(f"  {split}: {len(data)} 样本")
        
        # 保存全局配置
        global_config = {
            "processed_attack_types": list(results.keys()),
            "tokenizer_name": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "special_tokens": self.tokenizer.special_tokens_map,
            "max_length": max_length
        }
        
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(global_config, f, indent=2)
        
        print(f"\n✅ 全局配置保存到: {config_path}")
        print(f"✅ 所有数据集保存在: {self.output_dir}")
    
    def verify_dataset(self, attack_type: str):
        """验证处理后的数据集"""
        print(f"\n{'='*60}")
        print(f"验证 {attack_type.upper()} 数据集")
        print(f"{'='*60}")
        
        dataset_path = self.output_dir / attack_type
        if not dataset_path.exists():
            print(f"❌ 数据集不存在: {dataset_path}")
            return
        
        # 加载数据集
        dataset = DatasetDict.load_from_disk(str(dataset_path))
        
        print(f"\n数据集结构:")
        print(f"  Splits: {list(dataset.keys())}")
        print(f"  Features: {dataset['train'].features}")
        
        # 查看样本
        print(f"\n样本示例 (train):")
        for i in range(min(3, len(dataset['train']))):
            sample = dataset['train'][i]
            text = sample['text']
            input_ids = sample['input_ids']
            print(f"\n样本 {i+1}:")
            print(f"  文本: {text[:100]}...")
            print(f"  Token数: {len(input_ids)}")
            print(f"  前10个token: {input_ids[:10]}")
            
            # 解码验证
            decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            print(f"  解码: {decoded[:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="GPTFuzzer数据预处理 (按论文要求: SQLi/XSS=512K, RCE=37K)"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default="data/gptfuzzer",
                       help="输出目录")
    parser.add_argument("--model-name", type=str, default="gpt2",
                       help="Tokenizer模型名称")
    parser.add_argument("--max-length", type=int, default=128,
                       help="最大序列长度")
    parser.add_argument("--attack-type", type=str, choices=["sqli", "xss", "rce", "all"],
                       default="all", help="处理哪个攻击类型")
    parser.add_argument("--verify", action="store_true",
                       help="验证处理后的数据集")
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = GPTFuzzerDataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    if args.verify:
        # 验证模式
        if args.attack_type == "all":
            for attack_type in ["sqli", "xss", "rce"]:
                preprocessor.verify_dataset(attack_type)
        else:
            preprocessor.verify_dataset(args.attack_type)
    else:
        # 处理模式
        if args.attack_type == "all":
            preprocessor.create_all_datasets(max_length=args.max_length)
        else:
            preprocessor.create_dataset(args.attack_type, max_length=args.max_length)


if __name__ == "__main__":
    main()
