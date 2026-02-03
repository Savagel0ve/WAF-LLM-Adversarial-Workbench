"""
测试奖励模型 - 推理和评估
支持 Qwen2.5-Coder 和其他现代 LLM

使用训练好的奖励模型预测payload的WAF绕过概率
"""
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RewardModelInference:
    """奖励模型推理器 - 支持 Qwen2.5-Coder 和其他现代 LLM"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化推理器
        
        Args:
            model_path: 训练好的模型路径
            device: 设备 (cuda/cpu)
        """
        self.model_path = Path(model_path)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🔍 加载模型: {self.model_path}")
        logger.info(f"   设备: {self.device}")
        
        # 加载 tokenizer 和模型 (使用 Auto 类支持各种模型)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # 设置 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"   模型类型: {type(self.model).__name__}")
        logger.info("✅ 模型加载完成")
    
    def predict_single(self, payload: str, return_logits: bool = False) -> float:
        """
        预测单个payload的绕过概率
        
        Args:
            payload: 攻击payload
            return_logits: 是否返回原始logits
            
        Returns:
            绕过概率 [0, 1]
        """
        # Tokenize
        inputs = self.tokenizer(
            payload,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze()
        
        if return_logits:
            return logits.item()
        
        # 转换为概率
        prob = torch.sigmoid(logits).item()
        return prob
    
    def predict_batch(self, payloads: List[str], batch_size: int = 32) -> List[float]:
        """
        批量预测
        
        Args:
            payloads: payload列表
            batch_size: 批次大小
            
        Returns:
            概率列表
        """
        probs = []
        
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze()
            
            # 转换为概率
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            
            if len(batch) == 1:
                batch_probs = [batch_probs.item()]
            else:
                batch_probs = batch_probs.tolist()
            
            probs.extend(batch_probs)
        
        return probs
    
    def evaluate_payloads(self, payloads: List[str]) -> Dict:
        """
        评估一组payload
        
        Args:
            payloads: payload列表
            
        Returns:
            统计信息字典
        """
        logger.info(f"评估 {len(payloads)} 个payload...")
        
        probs = self.predict_batch(payloads)
        
        # 统计
        import numpy as np
        probs_array = np.array(probs)
        
        stats = {
            "count": len(probs),
            "mean": float(np.mean(probs_array)),
            "std": float(np.std(probs_array)),
            "min": float(np.min(probs_array)),
            "max": float(np.max(probs_array)),
            "median": float(np.median(probs_array)),
            "high_confidence_bypass": int(np.sum(probs_array > 0.8)),
            "low_confidence_bypass": int(np.sum(probs_array < 0.2)),
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="测试奖励模型")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--payload",
        type=str,
        help="单个payload (用于快速测试)"
    )
    parser.add_argument(
        "--payload_file",
        type=str,
        help="payload文件 (每行一个)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批次大小"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("奖励模型推理测试")
    logger.info("="*60)
    
    # 初始化推理器
    inferencer = RewardModelInference(args.model_path)
    
    # 单个payload测试
    if args.payload:
        logger.info(f"\n测试payload: {args.payload}")
        prob = inferencer.predict_single(args.payload)
        logger.info(f"绕过概率: {prob:.4f}")
        
        if prob > 0.8:
            logger.info("🟢 高概率绕过")
        elif prob > 0.5:
            logger.info("🟡 中等概率绕过")
        else:
            logger.info("🔴 低概率绕过")
    
    # 文件测试
    elif args.payload_file:
        logger.info(f"\n从文件加载payload: {args.payload_file}")
        
        with open(args.payload_file, 'r', encoding='utf-8') as f:
            payloads = [line.strip() for line in f if line.strip()]
        
        logger.info(f"加载 {len(payloads)} 个payload")
        
        # 评估
        stats = inferencer.evaluate_payloads(payloads)
        
        logger.info("\n" + "="*60)
        logger.info("评估结果:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 显示示例
        logger.info("\n" + "="*60)
        logger.info("示例预测 (前10个):")
        
        probs = inferencer.predict_batch(payloads[:10])
        for payload, prob in zip(payloads[:10], probs):
            status = "🟢" if prob > 0.8 else "🟡" if prob > 0.5 else "🔴"
            logger.info(f"{status} {prob:.4f} | {payload[:60]}")
    
    else:
        # 交互式测试
        logger.info("\n进入交互式测试模式 (输入 'quit' 退出)")
        
        while True:
            try:
                payload = input("\n请输入payload: ").strip()
                
                if payload.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not payload:
                    continue
                
                prob = inferencer.predict_single(payload)
                
                if prob > 0.8:
                    status = "🟢 高概率绕过"
                elif prob > 0.5:
                    status = "🟡 中等概率绕过"
                else:
                    status = "🔴 低概率绕过"
                
                logger.info(f"绕过概率: {prob:.4f} | {status}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"错误: {e}")
    
    logger.info("\n✅ 完成!")


if __name__ == "__main__":
    main()
