"""
统一实验运行器
==============

提供统一的接口来运行各种实验，管理模型加载、数据准备、评估流程。
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import (
    ExperimentConfig,
    get_model_path,
    get_data_path,
    ATTACK_TYPES,
    WAF_TYPES,
)
from experiments.metrics import (
    ExperimentMetrics,
    MetricsTracker,
    calculate_tp,
    calculate_er,
    calculate_nrr,
    calculate_ter,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """统一实验运行器"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        device: str = "cuda"
    ):
        """
        初始化运行器
        
        Args:
            config: 实验配置
            device: 运行设备
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 设置随机种子
        self.set_seed(config.seed)
        
        # 创建输出目录
        self.results_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config.save(self.results_dir / "config.json")
        
        logger.info(f"实验运行器初始化完成")
        logger.info(f"  实验名称: {config.experiment_name}")
        logger.info(f"  输出目录: {self.results_dir}")
        logger.info(f"  设备: {self.device}")
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"随机种子设置为: {seed}")
    
    def load_model(
        self,
        model_path: str,
        model_type: str = "causal_lm"
    ) -> Tuple[Any, Any]:
        """
        加载模型和tokenizer
        
        Args:
            model_path: 模型路径
            model_type: 模型类型 (causal_lm/sequence_classification)
            
        Returns:
            (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        
        logger.info(f"加载模型: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        elif model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"模型加载完成")
        return model, tokenizer
    
    def load_data(
        self,
        attack_type: str,
        data_type: str = "train",
        max_samples: Optional[int] = None
    ) -> List[str]:
        """
        加载数据
        
        Args:
            attack_type: 攻击类型
            data_type: 数据类型 (train/val/test)
            max_samples: 最大样本数
            
        Returns:
            数据列表
        """
        data_path = get_data_path(attack_type, data_type, self.config.data_dir)
        logger.info(f"加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f if line.strip()]
        
        if max_samples and max_samples < len(data):
            data = random.sample(data, max_samples)
        
        logger.info(f"数据加载完成: {len(data)} 条")
        return data
    
    def generate_payloads(
        self,
        model: Any,
        tokenizer: Any,
        num_samples: int,
        max_length: int = 128,
        temperature: float = 1.0,
        batch_size: int = 10,
        start_prompt: str = "",
    ) -> List[str]:
        """
        使用模型生成payload
        
        Args:
            model: 生成模型
            tokenizer: tokenizer
            num_samples: 生成数量
            max_length: 最大长度
            temperature: 温度
            batch_size: 批次大小
            start_prompt: 起始提示
            
        Returns:
            生成的payload列表
        """
        logger.info(f"生成 {num_samples} 个payload...")
        
        payloads = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="生成"):
                current_batch_size = min(batch_size, num_samples - len(payloads))
                
                # 准备输入
                if start_prompt:
                    prompt_ids = tokenizer.encode(start_prompt, add_special_tokens=False)
                else:
                    prompt_ids = [tokenizer.bos_token_id or tokenizer.eos_token_id]
                
                input_ids = torch.tensor([prompt_ids] * current_batch_size, device=self.device)
                attention_mask = torch.ones_like(input_ids)
                
                # 生成
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # 解码
                for output in outputs:
                    payload = tokenizer.decode(output, skip_special_tokens=True)
                    if start_prompt and payload.startswith(start_prompt):
                        payload = payload[len(start_prompt):].strip()
                    payloads.append(payload)
        
        return payloads[:num_samples]
    
    def test_waf(
        self,
        payloads: List[str],
        waf_url: str,
        waf_type: str = "modsecurity",
        param_name: str = "id",
        method: str = "get",
        timeout: int = 5,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        在WAF上测试payload
        
        Args:
            payloads: 要测试的payload列表
            waf_url: WAF URL
            waf_type: WAF类型
            param_name: 参数名
            method: HTTP方法
            timeout: 超时时间
            
        Returns:
            (绕过的payload, 被阻止的payload, 错误的payload)
        """
        import requests
        
        logger.info(f"测试 {len(payloads)} 个payload...")
        
        bypassed = []
        blocked = []
        errors = []
        
        for payload in tqdm(payloads, desc=f"WAF测试 ({waf_type})"):
            try:
                if method.lower() == "post":
                    response = requests.post(
                        waf_url,
                        data={param_name: payload},
                        timeout=timeout,
                    )
                else:
                    response = requests.get(
                        waf_url,
                        params={param_name: payload},
                        timeout=timeout,
                    )
                
                # 200表示绕过，403表示被阻止
                if response.status_code == 200:
                    bypassed.append(payload)
                else:
                    blocked.append(payload)
                    
            except requests.exceptions.Timeout:
                errors.append(payload)
            except Exception as e:
                errors.append(payload)
        
        logger.info(f"测试完成: 绕过={len(bypassed)}, 阻止={len(blocked)}, 错误={len(errors)}")
        return bypassed, blocked, errors
    
    def run_evaluation(
        self,
        attack_type: str,
        waf_type: str,
        method_name: str,
        generate_func: Callable,
        num_requests: int,
        checkpoints: Optional[List[int]] = None,
    ) -> ExperimentMetrics:
        """
        运行单次评估
        
        Args:
            attack_type: 攻击类型
            waf_type: WAF类型
            method_name: 方法名称
            generate_func: 生成函数 (batch_size) -> List[str]
            num_requests: 总请求数
            checkpoints: 检查点列表
            
        Returns:
            评估指标
        """
        logger.info(f"开始评估: {method_name} on {attack_type}/{waf_type}")
        
        waf_url = self.config.waf_urls.get(waf_type, "http://localhost:8001")
        
        # 初始化追踪器
        if checkpoints is None:
            checkpoints = [num_requests]
        tracker = MetricsTracker(checkpoints)
        tracker.start()
        
        # 生成和测试
        batch_size = 100
        all_payloads = []
        
        start_time = time.time()
        
        while len(all_payloads) < num_requests:
            # 生成一批
            current_batch_size = min(batch_size, num_requests - len(all_payloads))
            payloads = generate_func(current_batch_size)
            
            # 测试
            bypassed, blocked, errors = self.test_waf(
                payloads, waf_url, waf_type
            )
            
            # 记录结果
            for payload in payloads:
                tracker.add_result(payload, payload in bypassed)
            
            all_payloads.extend(payloads)
        
        end_time = time.time()
        
        # 汇总指标
        results = tracker.get_results()
        
        metrics = ExperimentMetrics(
            total_generated=len(all_payloads),
            unique_payloads=results["final"]["unique_payloads"],
            valid_payloads=len(all_payloads),  # 简化处理
            tested_payloads=len(all_payloads),
            bypassed=results["final"]["tp"],
            blocked=len(all_payloads) - results["final"]["tp"],
            errors=0,
            generation_time=end_time - start_time,
            testing_time=end_time - start_time,
            attack_type=attack_type,
            waf_type=waf_type,
            method=method_name,
        )
        
        return metrics, tracker.get_results()
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str,
        subdir: Optional[str] = None
    ):
        """
        保存结果
        
        Args:
            results: 结果字典
            filename: 文件名
            subdir: 子目录
        """
        if subdir:
            output_dir = self.results_dir / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.results_dir
        
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"结果已保存: {output_path}")
    
    def run_multiple(
        self,
        run_func: Callable,
        num_runs: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        运行多次实验取平均
        
        Args:
            run_func: 单次运行函数
            num_runs: 运行次数
            **kwargs: 传递给run_func的参数
            
        Returns:
            所有运行结果列表
        """
        if num_runs is None:
            num_runs = self.config.num_runs
        
        results = []
        for i in range(num_runs):
            logger.info(f"运行 {i+1}/{num_runs}")
            self.set_seed(self.config.seed + i)
            result = run_func(**kwargs)
            results.append(result)
        
        return results


class WarmupRunner(ExperimentRunner):
    """预热运行器，用于检查环境和配置"""
    
    def check_environment(self) -> Dict[str, bool]:
        """检查实验环境"""
        checks = {}
        
        # 检查CUDA
        checks["cuda_available"] = torch.cuda.is_available()
        if checks["cuda_available"]:
            checks["cuda_device"] = torch.cuda.get_device_name(0)
            checks["cuda_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        
        # 检查数据目录
        checks["data_dir_exists"] = os.path.exists(self.config.data_dir)
        
        # 检查各攻击类型数据
        for attack_type in ATTACK_TYPES:
            data_path = get_data_path(attack_type, "train", self.config.data_dir)
            checks[f"{attack_type}_data_exists"] = os.path.exists(data_path)
        
        # 检查模型目录
        checks["models_dir_exists"] = os.path.exists(self.config.models_dir)
        
        # 检查WAF连接
        import requests
        for waf_type, waf_url in self.config.waf_urls.items():
            try:
                response = requests.get(waf_url, timeout=5)
                checks[f"{waf_type}_waf_reachable"] = True
            except:
                checks[f"{waf_type}_waf_reachable"] = False
        
        # 打印检查结果
        logger.info("环境检查结果:")
        for key, value in checks.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")
        
        return checks
    
    def warmup(self, attack_type: str = "sqli", num_samples: int = 10) -> bool:
        """
        预热测试
        
        Args:
            attack_type: 测试的攻击类型
            num_samples: 测试样本数
            
        Returns:
            是否成功
        """
        logger.info(f"开始预热测试: {attack_type}")
        
        try:
            # 加载少量数据
            data = self.load_data(attack_type, "test", max_samples=num_samples)
            logger.info(f"  数据加载成功: {len(data)} 条")
            
            # 测试WAF
            for waf_type in self.config.waf_types:
                waf_url = self.config.waf_urls[waf_type]
                bypassed, blocked, errors = self.test_waf(
                    data[:min(5, len(data))], waf_url, waf_type
                )
                logger.info(f"  {waf_type} WAF测试: 绕过={len(bypassed)}, 阻止={len(blocked)}")
            
            logger.info("预热测试完成")
            return True
            
        except Exception as e:
            logger.error(f"预热测试失败: {e}")
            return False


if __name__ == "__main__":
    # 测试运行器
    from experiments.experiment_config import RQ1Config
    
    config = RQ1Config(
        output_dir="results_test",
        seed=42,
    )
    
    runner = WarmupRunner(config)
    runner.check_environment()
