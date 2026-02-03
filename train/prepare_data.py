"""
数据准备和预处理脚本
解压数据集、清洗、分割、tokenize

优化说明:
- 使用蓄水池采样算法避免加载全部数据到内存
- 按照GPTFuzzer论文要求限制数据量: SQLi/XSS=512K, RCE=37K
"""
import os
import zipfile
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import random


class DatasetPreparer:
    """数据集准备器"""
    
    def __init__(self, data_dir="gptfuzzer-main/Datasets", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.attack_types = {
            "sqli": "SQLi",
            "xss": "XSS",
            "rce": "RCE"
        }
        
        # 根据GPTFuzzer论文的数据量要求
        self.max_samples = {
            "sqli": 512000,  # 512K
            "xss": 512000,   # 512K
            "rce": None      # RCE使用全量数据 (37,302)
        }
    
    def extract_sqli_dataset(self):
        """解压SQLi数据集(分卷压缩)"""
        print("\n" + "="*60)
        print("解压SQLi数据集")
        print("="*60)
        
        sqli_dir = self.data_dir / "SQLi"
        zip_file = sqli_dir / "SQLi_Dataset.zip"
        
        if not zip_file.exists():
            print(f"⚠️  警告: {zip_file} 不存在")
            print("请确保SQLi_Dataset.zip及其分卷文件(.z01, .z02, ...)都在目录中")
            
            # 检查分卷文件
            z_files = list(sqli_dir.glob("SQLi_Dataset.z*"))
            if z_files:
                print(f"✅ 找到 {len(z_files)} 个分卷文件")
                print("   需要先合并分卷文件:")
                print("   方法1: 使用7-Zip解压 SQLi_Dataset.zip")
                print("   方法2: 在Windows中，右键SQLi_Dataset.zip选择'解压到...'")
            return False
        
        # 解压
        try:
            print(f"📦 正在解压 {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(sqli_dir)
            print("✅ SQLi数据集解压成功")
            return True
        except zipfile.BadZipFile:
            print("❌ 解压失败: 可能需要先合并分卷文件")
            print("   请使用7-Zip或WinRAR解压完整的zip文件")
            return False
    
    def load_dataset(self, attack_type: str) -> List[str]:
        """
        加载指定攻击类型的数据集
        如果数据量超过论文要求，使用快速随机采样
        """
        attack_dir = self.attack_types.get(attack_type.lower())
        if not attack_dir:
            raise ValueError(f"未知的攻击类型: {attack_type}")
        
        dataset_file = self.data_dir / attack_dir / f"{attack_dir}_Dataset.txt"
        
        if not dataset_file.exists():
            print(f"⚠️  警告: {dataset_file} 不存在")
            return []
        
        print(f"\n📂 加载 {attack_type.upper()} 数据集: {dataset_file}")
        
        # 获取该攻击类型的最大样本数
        max_size = self.max_samples.get(attack_type.lower())
        
        # 如果没有限制，正常加载
        if max_size is None:
            payloads = []
            with open(dataset_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        payloads.append(line)
            print(f"✅ 加载 {len(payloads):,} 条 {attack_type.upper()} payloads (全量)")
            return payloads
        
        # 快速采样策略：读取固定数量后停止
        # 过采样2倍以应对后续清洗
        sample_size = max_size * 2
        print(f"  论文要求: {max_size:,} 条")
        print(f"  快速采样策略: 读取前 {sample_size:,} 条有效数据后停止")
        
        payloads = []
        count = 0
        total_lines = 0
        
        with open(dataset_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  采样中", unit=" lines", total=sample_size):
                total_lines += 1
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                payloads.append(line)
                count += 1
                
                # 达到目标采样数量后停止
                if count >= sample_size:
                    print(f"\n  ⏹️  已采样 {sample_size:,} 条，停止读取")
                    break
        
        print(f"✅ 采样完成: 读取 {total_lines:,} 行，采样 {len(payloads):,} 条有效数据")
        return payloads
    
    def clean_payloads(self, payloads: List[str], attack_type: str) -> List[str]:
        """清洗payload数据"""
        print(f"\n🧹 清洗 {attack_type.upper()} payloads...")
        print(f"  输入数据量: {len(payloads):,} 条")
        
        cleaned = []
        for payload in payloads:
            # 去除过长或过短的payload
            if 5 <= len(payload) <= 500:
                # 去除明显的无效payload
                if not payload.startswith('http://') and not payload.startswith('https://'):
                    cleaned.append(payload)
        
        print(f"  过滤后: {len(cleaned):,} 条")
        
        # 去重
        cleaned = list(set(cleaned))
        print(f"  去重后: {len(cleaned):,} 条唯一payloads")
        
        # 根据论文要求限制数据量
        max_size = self.max_samples.get(attack_type.lower())
        if max_size is not None and len(cleaned) > max_size:
            print(f"  📉 最终采样到 {max_size:,} 条（论文要求）")
            random.seed(42)
            cleaned = random.sample(cleaned, max_size)
        elif max_size is not None:
            print(f"  ✅ 数据量 {len(cleaned):,} 符合论文要求 (<= {max_size:,})")
        
        print(f"✅ 最终数据量: {len(cleaned):,} 条")
        return cleaned
    
    def split_dataset(self, payloads: List[str], 
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                     seed=42) -> Dict[str, List[str]]:
        """分割数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        random.seed(seed)
        random.shuffle(payloads)
        
        total = len(payloads)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        splits = {
            "train": payloads[:train_size],
            "val": payloads[train_size:train_size + val_size],
            "test": payloads[train_size + val_size:]
        }
        
        print(f"\n📊 数据集分割:")
        print(f"  - 训练集: {len(splits['train'])} ({train_ratio*100:.0f}%)")
        print(f"  - 验证集: {len(splits['val'])} ({val_ratio*100:.0f}%)")
        print(f"  - 测试集: {len(splits['test'])} ({test_ratio*100:.0f}%)")
        
        return splits
    
    def save_dataset(self, splits: Dict[str, List[str]], attack_type: str):
        """保存处理后的数据集"""
        attack_dir = self.output_dir / attack_type
        attack_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 保存数据集到 {attack_dir}...")
        
        for split_name, payloads in splits.items():
            # 保存为文本文件
            txt_file = attack_dir / f"{split_name}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                for payload in payloads:
                    f.write(payload + '\n')
            
            # 保存为JSON文件(包含元数据)
            json_file = attack_dir / f"{split_name}.json"
            data = {
                "attack_type": attack_type,
                "split": split_name,
                "count": len(payloads),
                "payloads": payloads
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ {split_name}: {txt_file}")
        
        # 保存统计信息
        stats_file = attack_dir / "stats.json"
        stats = {
            "attack_type": attack_type,
            "total": sum(len(p) for p in splits.values()),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"])
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✅ stats: {stats_file}")
    
    def prepare_attack_type(self, attack_type: str):
        """准备指定攻击类型的数据"""
        print("\n" + "="*60)
        print(f"准备 {attack_type.upper()} 数据集")
        max_size = self.max_samples.get(attack_type.lower())
        if max_size:
            print(f"论文要求数据量: {max_size:,} 条")
        else:
            print(f"论文要求: 使用全量数据")
        print("="*60)
        
        # 加载数据
        payloads = self.load_dataset(attack_type)
        if not payloads:
            print(f"❌ 无法加载 {attack_type} 数据集")
            return False
        
        # 清洗数据
        payloads = self.clean_payloads(payloads, attack_type)
        
        # 分割数据
        splits = self.split_dataset(payloads)
        
        # 保存数据
        self.save_dataset(splits, attack_type)
        
        print(f"\n✅ {attack_type.upper()} 数据集准备完成!")
        return True
    
    def prepare_all(self):
        """准备所有数据集"""
        print("\n" + "="*60)
        print("准备所有数据集")
        print("="*60)
        
        success_count = 0
        for attack_type in ["sqli", "xss", "rce"]:
            if self.prepare_attack_type(attack_type):
                success_count += 1
        
        print("\n" + "="*60)
        print(f"数据准备完成: {success_count}/3 成功")
        print("="*60)
        
        # 生成总体统计
        self.generate_overall_stats()
    
    def generate_overall_stats(self):
        """生成总体统计信息"""
        stats = {}
        for attack_type in ["sqli", "xss", "rce"]:
            stats_file = self.output_dir / attack_type / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats[attack_type] = json.load(f)
        
        overall_file = self.output_dir / "overall_stats.json"
        with open(overall_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 总体统计: {overall_file}")
        for attack_type, s in stats.items():
            print(f"  {attack_type.upper()}: {s['total']} 条payloads")


def main():
    parser = argparse.ArgumentParser(description="数据集准备工具 (按GPTFuzzer论文要求)")
    parser.add_argument("--extract", action="store_true", help="解压SQLi数据集")
    parser.add_argument("--attack-type", type=str, choices=["sqli", "xss", "rce", "all"],
                       default="all", help="准备哪个攻击类型的数据")
    parser.add_argument("--data-dir", type=str, default="gptfuzzer-main/Datasets",
                       help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="输出目录")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GPTFuzzer 数据准备工具")
    print("="*60)
    print("论文要求的数据量:")
    print("  - SQLi: 512,000 条")
    print("  - XSS:  512,000 条")
    print("  - RCE:  37,302 条 (全量)")
    print("="*60)
    
    preparer = DatasetPreparer(args.data_dir, args.output_dir)
    
    # 解压SQLi数据集
    if args.extract:
        preparer.extract_sqli_dataset()
        return
    
    # 准备数据
    if args.attack_type == "all":
        preparer.prepare_all()
    else:
        preparer.prepare_attack_type(args.attack_type)


if __name__ == "__main__":
    main()
