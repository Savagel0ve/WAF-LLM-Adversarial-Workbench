"""
生成标记数据 - 通过WAF测试payload并打标签

根据GPTFuzzer论文:
- SQLi: 4000条
- XSS/RCE: 2000条
"""
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import random
from tqdm import tqdm
import time
import logging

from waf_env import WAFEnvironment
from verifier import UniversalVerifier


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_payloads_from_file(file_path: Path, max_samples: int = None) -> List[str]:
    """从文件加载payload"""
    payloads = []
    
    # 支持txt和json格式
    if file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            payloads = [line.strip() for line in f if line.strip()]
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 假设json格式是列表或字典
            if isinstance(data, list):
                payloads = data
            elif isinstance(data, dict) and 'data' in data:
                payloads = data['data']
    
    # 随机采样
    if max_samples and len(payloads) > max_samples:
        payloads = random.sample(payloads, max_samples)
    
    return payloads


def test_payloads_with_waf(
    payloads: List[str],
    waf_env: WAFEnvironment,
    verifier: UniversalVerifier,
    attack_type: str,
    rate_limit_delay: float = 0.1
) -> List[Dict]:
    """
    使用WAF测试payload并生成标记数据
    
    Args:
        payloads: payload列表
        waf_env: WAF环境
        verifier: 验证器
        attack_type: 攻击类型
        rate_limit_delay: 请求间隔(秒)
        
    Returns:
        标记数据列表 [{text, label, status_code, response_time}]
    """
    labeled_data = []
    
    logger.info(f"开始测试 {len(payloads)} 个payload...")
    
    for payload in tqdm(payloads, desc=f"测试{attack_type.upper()}"):
        try:
            # 验证payload
            verify_result = verifier.verify(payload, attack_type)
            
            # 跳过无效payload
            if not verify_result["syntax_valid"]:
                continue
            
            # 发送到WAF
            waf_response = waf_env.send_payload(payload, attack_type)
            
            # 生成标签
            # Label 1: 绕过WAF (未被拦截)
            # Label 0: 被拦截
            label = 0 if waf_response.blocked else 1
            
            labeled_data.append({
                "text": payload,
                "label": label,
                "status_code": waf_response.status_code,
                "response_time": waf_response.response_time,
                "syntax_valid": verify_result["syntax_valid"],
                "executable": verify_result["executable"]
            })
            
            # 速率限制
            time.sleep(rate_limit_delay)
            
        except Exception as e:
            logger.error(f"测试payload失败: {payload[:50]}... - {e}")
            continue
    
    logger.info(f"成功标记 {len(labeled_data)} 个payload")
    
    return labeled_data


def balance_dataset(labeled_data: List[Dict], balance_ratio: float = 0.5) -> List[Dict]:
    """
    平衡数据集 - 确保正负样本比例
    
    Args:
        labeled_data: 标记数据
        balance_ratio: 正样本比例 (0.5 = 1:1)
        
    Returns:
        平衡后的数据
    """
    # 分离正负样本
    positive_samples = [d for d in labeled_data if d["label"] == 1]
    negative_samples = [d for d in labeled_data if d["label"] == 0]
    
    logger.info(f"原始数据: 正样本={len(positive_samples)}, 负样本={len(negative_samples)}")
    
    # 计算目标数量
    total_count = len(labeled_data)
    target_positive = int(total_count * balance_ratio)
    target_negative = total_count - target_positive
    
    # 采样
    if len(positive_samples) > target_positive:
        positive_samples = random.sample(positive_samples, target_positive)
    if len(negative_samples) > target_negative:
        negative_samples = random.sample(negative_samples, target_negative)
    
    # 合并
    balanced_data = positive_samples + negative_samples
    random.shuffle(balanced_data)
    
    logger.info(f"平衡后数据: 正样本={len(positive_samples)}, 负样本={len(negative_samples)}")
    
    return balanced_data


def split_dataset(
    labeled_data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, List[Dict]]:
    """
    划分数据集
    
    Args:
        labeled_data: 标记数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    random.shuffle(labeled_data)
    
    total = len(labeled_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": labeled_data[:train_end],
        "val": labeled_data[train_end:val_end],
        "test": labeled_data[val_end:]
    }
    
    logger.info(f"数据集划分: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits


def save_dataset(splits: Dict[str, List[Dict]], output_dir: Path, attack_type: str):
    """保存数据集"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, data in splits.items():
        # 保存为CSV (用于训练)
        csv_path = output_dir / f"{attack_type}_{split_name}.csv"
        df = pd.DataFrame(data)
        df[["text", "label"]].to_csv(csv_path, index=False)
        logger.info(f"保存 {split_name} 到 {csv_path}")
        
        # 保存完整JSON (包含额外信息)
        json_path = output_dir / f"{attack_type}_{split_name}_full.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 保存统计信息
    stats = {
        "attack_type": attack_type,
        "total_samples": sum(len(splits[s]) for s in splits),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "positive_ratio": {
            split: sum(1 for d in data if d["label"] == 1) / len(data) if data else 0
            for split, data in splits.items()
        }
    }
    
    stats_path = output_dir / f"{attack_type}_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"保存统计信息到 {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="生成WAF绕过标记数据")
    parser.add_argument(
        "--attack_type",
        type=str,
        required=True,
        choices=["sqli", "xss", "rce"],
        help="攻击类型"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入payload文件 (txt或json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/labeled",
        help="输出目录"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="采样数量 (默认: SQLi=4000, XSS/RCE=2000)"
    )
    parser.add_argument(
        "--waf_url",
        type=str,
        default="http://localhost:8081",
        help="WAF URL"
    )
    parser.add_argument(
        "--balance_ratio",
        type=float,
        default=0.5,
        help="正样本比例 (0-1)"
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=0.1,
        help="请求间隔(秒)"
    )
    
    args = parser.parse_args()
    
    # 设置默认采样数量
    if args.num_samples is None:
        args.num_samples = 4000 if args.attack_type == "sqli" else 2000
    
    logger.info("="*60)
    logger.info("生成标记数据")
    logger.info("="*60)
    logger.info(f"攻击类型: {args.attack_type}")
    logger.info(f"采样数量: {args.num_samples}")
    logger.info(f"WAF URL: {args.waf_url}")
    
    # 加载payload
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return
    
    payloads = load_payloads_from_file(input_path, args.num_samples)
    logger.info(f"加载 {len(payloads)} 个payload")
    
    # 初始化环境
    waf_env = WAFEnvironment(
        waf_type="modsecurity",
        modsecurity_url=args.waf_url
    )
    
    # 测试连接
    if not waf_env.test_connection():
        logger.error("WAF连接失败!")
        return
    
    verifier = UniversalVerifier()
    
    # 测试并标记
    labeled_data = test_payloads_with_waf(
        payloads,
        waf_env,
        verifier,
        args.attack_type,
        args.rate_limit
    )
    
    if not labeled_data:
        logger.error("没有生成任何标记数据!")
        return
    
    # 平衡数据集
    balanced_data = balance_dataset(labeled_data, args.balance_ratio)
    
    # 划分数据集
    splits = split_dataset(balanced_data)
    
    # 保存
    output_dir = Path(args.output_dir)
    save_dataset(splits, output_dir, args.attack_type)
    
    # 打印WAF统计
    logger.info("\n" + "="*60)
    logger.info("WAF测试统计:")
    stats = waf_env.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n✅ 完成!")


if __name__ == "__main__":
    main()
