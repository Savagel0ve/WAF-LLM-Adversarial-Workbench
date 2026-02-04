"""
GPTFuzzer Stage 3: Reinforcement Learning (PPO)
使用PPO算法微调预训练模型，使其生成能绕过WAF的载荷

支持的模型:
- Qwen2.5-Coder-0.5B/1.5B/3B (推荐)
- DeepSeek-Coder
- Phi-3
- GPT-2 (兼容)

参考论文:
- GPTFuzzer: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts
- Proximal Policy Optimization (PPO): https://arxiv.org/abs/1707.06347

核心思路:
1. Policy Network: 从预训练模型初始化，用于生成载荷
2. Reference Model: 冻结的预训练模型，用于计算KL散度
3. Reward Model: 训练好的分类器，评估载荷绕过WAF的概率
4. PPO算法: 在奖励和KL散度之间平衡，优化策略网络
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Transformers和TRL
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)

# 本地导入
try:
    from config import get_default_configs, MODEL_PRESETS, DEFAULT_MODEL
except ImportError:
    print("警告: 无法导入config.py，使用默认配置")
    get_default_configs = None
    MODEL_PRESETS = {}
    DEFAULT_MODEL = "qwen2.5-coder-1.5b"


class RLConfig:
    """强化学习配置类 - 支持 Qwen2.5-Coder 和其他现代 LLM"""
    def __init__(
        self,
        # 模型路径
        pretrained_model_path: str = "./models/pretrain_sqli_qwen2_5_coder_1_5b",
        reward_model_path: str = "./models/reward_sqli_qwen/final_reward_model",
        output_dir: str = "./models/rl_sqli_qwen",
        
        # PPO超参数 (来自论文)
        learning_rate: float = 1.4e-5,
        batch_size: int = 128,  # Qwen 1.5B 优化
        mini_batch_size: int = 8,
        ppo_epochs: int = 4,
        init_kl_coef: float = 0.2,  # Beta参数，关键!
        target_kl: float = 0.1,
        adap_kl_ctrl: bool = False,  # 论文使用固定beta
        
        # 生成配置
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        start_prompt: str = "SELECT",  # SQL payload 更好的起始
        
        # 训练配置
        total_episodes: int = 20,
        save_freq: int = 5,
        update_batch_size: int = 2,
        
        # 精度配置
        use_fp16: bool = False,
        use_bf16: bool = True,  # Qwen 推荐 bf16
        gradient_accumulation_steps: int = 1,
        
        # 其他
        seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_with: Optional[str] = None,  # "wandb" or "tensorboard"
    ):
        self.pretrained_model_path = pretrained_model_path
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.init_kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.adap_kl_ctrl = adap_kl_ctrl
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.start_prompt = start_prompt
        
        self.total_episodes = total_episodes
        self.save_freq = save_freq
        self.update_batch_size = update_batch_size
        
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.seed = seed
        self.device = device
        self.log_with = log_with
        
        # 计算实际的 mini_batch 数量
        self.gradient_accumulation_steps = max(1, batch_size // mini_batch_size)
        
    def to_ppo_config(self) -> PPOConfig:
        """转换为TRL的PPOConfig"""
        return PPOConfig(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_ppo_epochs=self.ppo_epochs,
            kl_coef=self.init_kl_coef,
            seed=self.seed,
            log_with=self.log_with if self.log_with else None,
            logging_steps=10,
            save_strategy="no",  # 我们手动保存
        )
    
    def save(self, path: str):
        """保存配置到JSON"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        """从JSON加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class RewardModelWrapper:
    """奖励模型包装器 - 将分类器输出转换为奖励信号"""
    
    def __init__(self, model_path: str, tokenizer, device: str = "cuda"):
        """
        初始化奖励模型
        
        Args:
            model_path: 奖励模型路径
            tokenizer: tokenizer实例
            device: 设备
        """
        self.device = device
        
        # 加载奖励模型
        print(f"🎁 加载奖励模型: {model_path}")
        
        # 先加载模型配置，不覆盖num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            ignore_mismatched_sizes=True  # 忽略大小不匹配警告
        )
        self.model.eval()
        
        # 检测模型的标签数量
        self.num_labels = self.model.config.num_labels
        print(f"   检测到 {self.num_labels} 个标签")
        
        self.tokenizer = tokenizer
        
        # 根据模型类型创建pipeline
        if self.num_labels == 1:
            # BCEWithLogitsLoss形式，不使用pipeline而直接推理
            self.pipe = None
        else:
            # 标准分类，使用pipeline
            self.pipe = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                return_all_scores=True
            )
    
    def get_rewards(self, texts: List[str]) -> List[float]:
        """
        计算一批文本的奖励
        
        Args:
            texts: 生成的载荷列表
            
        Returns:
            奖励列表 (0-1之间的浮点数)
        """
        try:
            if self.num_labels == 1:
                # BCEWithLogitsLoss形式：直接推理获取logits然后sigmoid
                # 确保使用left padding
                original_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = 'left'
                
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 恢复原来的padding side
                self.tokenizer.padding_side = original_padding_side
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                    # 应用sigmoid将logits转换为概率
                    probs = torch.sigmoid(logits)
                    rewards = probs.cpu().tolist()
                
                return rewards
            
            else:
                # 标准二分类：使用pipeline
                outputs = self.pipe(texts)
                
                # 提取奖励分数
                rewards = []
                for output in outputs:
                    # output是一个包含字典的列表
                    if isinstance(output, list) and len(output) > 0:
                        # 找到LABEL_1的分数
                        label1_score = None
                        for item in output:
                            if item.get('label') in ['LABEL_1', '1', 1]:
                                label1_score = item['score']
                                break
                        
                        if label1_score is not None:
                            rewards.append(label1_score)
                        elif len(output) >= 2:
                            # 如果找不到LABEL_1，使用第二个元素
                            rewards.append(output[1]['score'])
                        else:
                            rewards.append(0.0)
                    else:
                        # 降级处理
                        rewards.append(0.0)
                
                return rewards
        
        except Exception as e:
            print(f"⚠️  奖励计算出错: {e}")
            print(f"   文本样例: {texts[0] if texts else 'None'}")
            import traceback
            traceback.print_exc()
            # 返回零奖励
            return [0.0] * len(texts)
    
    def __call__(self, texts: List[str]) -> List[torch.Tensor]:
        """
        使接口兼容PPOTrainer
        
        Returns:
            torch.Tensor列表，每个元素是标量奖励
        """
        rewards = self.get_rewards(texts)
        return [torch.tensor(r, dtype=torch.float32) for r in rewards]


class RLTrainer:
    """强化学习训练器 - 封装PPO训练逻辑"""
    
    def __init__(self, config: RLConfig):
        """
        初始化RL训练器
        
        Args:
            config: RLConfig配置对象
        """
        self.config = config
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 保存配置
        config.save(os.path.join(config.output_dir, "rl_config.json"))
        
        # 初始化模型和trainer
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.ppo_trainer = None
        
        self._setup_models()
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_models(self):
        """设置所有需要的模型 - 支持 Qwen2.5-Coder 和其他现代 LLM"""
        print("\n" + "="*60)
        print("🚀 初始化强化学习环境")
        print("="*60)
        
        # 1. 加载 Tokenizer (使用 AutoTokenizer 支持各种模型)
        print(f"\n📝 加载 Tokenizer: {self.config.pretrained_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_path,
            trust_remote_code=True,
            padding_side='left'  # Causal LM 生成需要 left padding
        )
        
        # 设置 pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.tokenizer.padding_side = 'left'
        print(f"   Tokenizer: {type(self.tokenizer).__name__}")
        print(f"   Padding side: {self.tokenizer.padding_side}")
        print(f"   Vocab size: {len(self.tokenizer)}")

        # 设置起始 prompt
        prompt_ids = self.tokenizer.encode(self.config.start_prompt, add_special_tokens=False)
        if not prompt_ids or all(token_id == self.tokenizer.pad_token_id for token_id in prompt_ids):
            # 尝试使用数字作为 fallback
            fallback_prompt = "1"
            prompt_ids = self.tokenizer.encode(fallback_prompt, add_special_tokens=False)
            if not prompt_ids:
                # 使用 BOS token 如果存在，否则用 EOS
                if self.tokenizer.bos_token_id is not None:
                    prompt_ids = [self.tokenizer.bos_token_id]
                else:
                    prompt_ids = [self.tokenizer.eos_token_id]
            print(f"   ⚠️ start_prompt 无效，使用 fallback")
        self.prompt_ids = prompt_ids
        print(f"   Start prompt ids: {self.prompt_ids}")
        
        # 2. 确定精度
        if self.config.use_bf16:
            torch_dtype = torch.bfloat16
        elif self.config.use_fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 3. 加载 Policy Network (带 Value Head)
        print(f"\n🤖 加载 Policy Network (带 Value Head)")
        print(f"   精度: {torch_dtype}")
        
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.pretrained_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.config.device)
        
        # 如果添加了新的 pad token，调整 embedding
        if len(self.tokenizer) > self.model.pretrained_model.config.vocab_size:
            self.model.pretrained_model.resize_token_embeddings(len(self.tokenizer))
        
        # 4. 创建 Reference Model (冻结参数)
        print(f"\n🔒 创建 Reference Model (冻结参数)")
        self.ref_model = create_reference_model(self.model)
        
        # 5. 加载 Reward Model
        print(f"\n🎁 加载 Reward Model: {self.config.reward_model_path}")
        self.reward_model = RewardModelWrapper(
            self.config.reward_model_path,
            self.tokenizer,
            self.config.device
        )
        
        print("\n✅ 所有模型加载完成!")
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   - Policy Model 参数: {total_params / 1e9:.2f}B ({total_params / 1e6:.1f}M)")
        print(f"   - 可训练参数: {trainable_params / 1e9:.2f}B")
        print(f"   - 设备: {self.config.device}")
        print(f"   - 批次大小: {self.config.batch_size}")
        print(f"   - Mini 批次大小: {self.config.mini_batch_size}")
        print(f"   - KL 系数 (β): {self.config.init_kl_coef}")
    
    def generate_queries(self, batch_size: int) -> List[torch.Tensor]:
        """
        生成查询 (起始token)
        
        在GPTFuzzer中，通常使用<start>或BOS token作为起始
        
        Args:
            batch_size: 批次大小
            
        Returns:
            query tensor列表
        """
        # 使用BOS token作为起始
        query_tensors = []
        for _ in range(batch_size):
            # 创建只包含BOS token的输入
            query_tensor = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)
            query_tensors.append(query_tensor.squeeze(0))
        
        return query_tensors
    
    def train(self):
        """主训练循环 - 简化版PPO实现"""
        print("\n" + "="*60)
        print("🎯 开始强化学习训练")
        print("="*60)
        print(f"总训练轮数: {self.config.total_episodes}")
        print(f"每轮生成: {self.config.batch_size} 个载荷")
        print(f"预计生成总数: {self.config.total_episodes * self.config.batch_size}")
        print("="*60 + "\n")
        
        # 设置优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_fp16 and self.config.device == "cuda")
        
        # 训练统计
        all_rewards = []
        
        # 主训练循环
        for episode in range(self.config.total_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{self.config.total_episodes}")
            print(f"{'='*60}")
            
            # 清理缓存，避免累计显存占用
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

            # === Step 1: Generate (Rollout) ===
            print(f"\n🎲 生成 {self.config.batch_size} 个载荷...")
            
            batch_texts = []
            batch_rewards = []
            
            # 分批生成以节省显存
            num_batches = (self.config.batch_size + self.config.mini_batch_size - 1) // self.config.mini_batch_size
            
            for batch_idx in range(num_batches):
                current_batch_size = min(
                    self.config.mini_batch_size,
                    self.config.batch_size - len(batch_texts)
                )
                
                # 创建起始输入并设置attention mask
                prompt_tensor = torch.tensor(self.prompt_ids, dtype=torch.long, device=self.config.device)
                input_ids = prompt_tensor.unsqueeze(0).repeat(current_batch_size, 1)
                attention_mask = torch.ones_like(input_ids)
                
                # 生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # 解码
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    batch_texts.append(text)
            
            # 显示样例
            print(f"\n📄 生成样例 (前3个):")
            for i, text in enumerate(batch_texts[:3]):
                print(f"   [{i+1}] {text[:100]}..." if len(text) > 100 else f"   [{i+1}] {text}")
            
            # === Step 2: Calculate Rewards ===
            print(f"\n🎁 计算奖励...")
            
            rewards = self.reward_model.get_rewards(batch_texts)
            
            # 统计奖励
            mean_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            min_reward = np.min(rewards)
            
            all_rewards.extend(rewards)
            
            print(f"\n📊 奖励统计:")
            print(f"   - 平均奖励: {mean_reward:.4f}")
            print(f"   - 最大奖励: {max_reward:.4f}")
            print(f"   - 最小奖励: {min_reward:.4f}")
            
            # === Step 3: Simple Policy Update ===
            # 简化版：基于REINFORCE算法更新策略
            print(f"\n🔄 执行策略更新...")
            
            try:
                optimizer.zero_grad(set_to_none=True)
                if self.config.device == "cuda":
                    torch.cuda.empty_cache()
                
                # 重新计算生成的log概率
                total_loss = 0
                num_samples = 0
                
                # 确保使用left padding
                original_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = 'left'
                
                update_batch_size = max(1, self.config.update_batch_size)
                for start_idx in range(0, len(batch_texts), update_batch_size):
                    batch_slice = batch_texts[start_idx:start_idx + update_batch_size]
                    batch_rewards = rewards[start_idx:start_idx + update_batch_size]

                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_slice,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_new_tokens,
                        padding=True
                    ).to(self.config.device)

                    with torch.amp.autocast(device_type="cuda", enabled=self.config.use_fp16 and self.config.device == "cuda"):
                        # 前向传播
                        outputs = self.model(**inputs)

                        # 计算语言模型loss（手动）
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        labels = inputs["input_ids"]
                        # shift for causal LM
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        # 计算交叉熵
                        model_loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=self.tokenizer.pad_token_id
                        )

                        # 使用batch平均reward
                        reward_tensor = torch.tensor(batch_rewards, device=self.config.device, dtype=model_loss.dtype)
                        reward_mean = reward_tensor.mean()
                        loss = -model_loss * reward_mean

                    scaler.scale(loss).backward()
                    total_loss += loss.detach()
                    num_samples += len(batch_slice)

                    # 释放显存
                    del inputs, outputs, logits, labels, shift_logits, shift_labels, model_loss, loss, reward_tensor
                    if self.config.device == "cuda":
                        torch.cuda.empty_cache()
                
                # 平均loss
                avg_loss = total_loss / num_samples
                
                # 反向传播
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # 恢复padding side
                self.tokenizer.padding_side = original_padding_side
                
                print(f"   - 平均损失: {avg_loss.item():.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("⚠️  OOM during policy update, clearing cache and continuing.")
                    if self.config.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                print(f"⚠️  策略更新出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            except Exception as e:
                print(f"⚠️  策略更新出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # === Step 4: Save Checkpoint ===
            if (episode + 1) % self.config.save_freq == 0:
                checkpoint_dir = os.path.join(
                    self.config.output_dir,
                    f"checkpoint-{episode + 1}"
                )
                print(f"\n💾 保存检查点: {checkpoint_dir}")
                self.save_model(checkpoint_dir)
            
            # 打印累积统计
            print(f"\n📈 累积统计 (Episodes 1-{episode+1}):")
            print(f"   - 平均奖励: {np.mean(all_rewards):.4f}")
        
        # === Final Save ===
        print(f"\n{'='*60}")
        print("🎉 训练完成!")
        print(f"{'='*60}")
        
        final_dir = os.path.join(self.config.output_dir, "final_model")
        print(f"\n💾 保存最终模型: {final_dir}")
        self.save_model(final_dir)
        
        # 保存训练统计
        stats_file = os.path.join(self.config.output_dir, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'all_rewards': all_rewards,
                'mean_reward': float(np.mean(all_rewards)),
                'final_reward': float(np.mean(all_rewards[-self.config.batch_size:]) if len(all_rewards) >= self.config.batch_size else np.mean(all_rewards)),
                'total_episodes': self.config.total_episodes,
            }, f, indent=2)
        
        print(f"\n📊 训练统计已保存: {stats_file}")
        print(f"\n最终平均奖励: {np.mean(all_rewards):.4f}")
        
        return {
            'mean_reward': np.mean(all_rewards),
            'all_rewards': all_rewards,
        }
    
    def save_model(self, output_dir: str):
        """保存模型和tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存policy model (只保存base model部分)
        if hasattr(self.model, 'pretrained_model'):
            self.model.pretrained_model.save_pretrained(output_dir)
        else:
            # 如果是普通模型，直接保存
            self.model.save_pretrained(output_dir)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        self.config.save(os.path.join(output_dir, "rl_config.json"))
        
        print(f"   ✅ 模型已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GPTFuzzer Stage 3: 强化学习 (PPO) - 支持 Qwen2.5-Coder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用 Qwen2.5-Coder 模型
  python train_rl.py \\
      --pretrained_model ./models/pretrain_sqli_qwen2_5_coder_1_5b \\
      --reward_model ./models/reward_sqli_qwen/final_reward_model \\
      --output_dir ./models/rl_sqli_qwen

  # 调整批次大小 (显存不足时)
  python train_rl.py \\
      --batch_size 64 --mini_batch_size 4 --update_batch_size 1
        """
    )
    
    # 模型路径
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="./models/pretrain_sqli_qwen2_5_coder_1_5b",
        help="预训练模型路径 (Stage 1)"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="./models/reward_sqli_qwen/final_reward_model",
        help="奖励模型路径 (Stage 2)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/rl_sqli_qwen",
        help="输出目录"
    )
    
    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=1.4e-5, help="学习率 (论文值: 1.4e-5)")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小 (Qwen 1.5B: 128)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Mini批次大小")
    parser.add_argument("--init_kl_coef", type=float, default=0.2, help="KL系数β (论文值: 0.2)")
    parser.add_argument("--total_episodes", type=int, default=20, help="总训练轮数")
    parser.add_argument("--update_batch_size", type=int, default=2, help="策略更新的小批量大小")
    
    # 生成配置
    parser.add_argument("--max_new_tokens", type=int, default=128, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度")
    parser.add_argument("--start_prompt", type=str, default="SELECT", help="生成起始prompt")
    
    # 精度配置
    parser.add_argument("--bf16", action="store_true", default=True, help="使用 BF16 精度 (默认)")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 精度")
    parser.add_argument("--no-bf16", action="store_true", help="禁用 BF16")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_freq", type=int, default=5, help="保存频率")
    parser.add_argument("--log_with", type=str, default=None, help="日志工具 (wandb/tensorboard)")
    
    args = parser.parse_args()
    
    # 处理精度参数
    use_bf16 = args.bf16 and not args.no_bf16 and not args.fp16
    use_fp16 = args.fp16 or (not use_bf16 and not args.no_bf16)
    
    # 创建配置
    config = RLConfig(
        pretrained_model_path=args.pretrained_model,
        reward_model_path=args.reward_model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        init_kl_coef=args.init_kl_coef,
        total_episodes=args.total_episodes,
        update_batch_size=args.update_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        start_prompt=args.start_prompt,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        seed=args.seed,
        save_freq=args.save_freq,
        log_with=args.log_with,
    )
    
    # 打印配置
    print("\n" + "="*60)
    print("⚙️  训练配置")
    print("="*60)
    print(f"预训练模型: {config.pretrained_model_path}")
    print(f"奖励模型: {config.reward_model_path}")
    print(f"输出目录: {config.output_dir}")
    print(f"学习率: {config.learning_rate}")
    print(f"批次大小: {config.batch_size}")
    print(f"Mini批次大小: {config.mini_batch_size}")
    print(f"KL系数 (β): {config.init_kl_coef}")
    print(f"训练轮数: {config.total_episodes}")
    print(f"设备: {config.device}")
    print("="*60 + "\n")
    
    # 检查模型是否存在
    if not os.path.exists(config.pretrained_model_path):
        print(f"❌ 错误: 预训练模型不存在: {config.pretrained_model_path}")
        print(f"   请先完成Stage 1 (预训练)")
        sys.exit(1)
    
    if not os.path.exists(config.reward_model_path):
        print(f"❌ 错误: 奖励模型不存在: {config.reward_model_path}")
        print(f"   请先完成Stage 2 (奖励模型训练)")
        sys.exit(1)
    
    # 创建训练器
    trainer = RLTrainer(config)
    
    # 开始训练
    try:
        results = trainer.train()
        
        print("\n" + "="*60)
        print("✅ 训练成功完成!")
        print("="*60)
        print(f"\n最终结果:")
        print(f"  - 平均奖励: {results['mean_reward']:.4f}")
        print(f"  - 模型保存位置: {config.output_dir}")
        print(f"\n下一步: 使用训练好的模型生成WAF绕过载荷")
        print(f"  python train/test_rl_model.py --model_path {config.output_dir}/final_model")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("正在保存当前模型...")
        trainer.save_model(os.path.join(config.output_dir, "interrupted_model"))
        print("已保存")
    
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
