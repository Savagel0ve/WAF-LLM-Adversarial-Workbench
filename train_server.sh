#!/bin/bash
# ============================================================
# GPTFuzzer 一键训练脚本 - Ubuntu 服务器版
# 针对 RTX 5090 32GB 优化
# ============================================================

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
CONDA_ENV_NAME="gptfuzzer"
PYTHON_VERSION="3.10"
ATTACK_TYPE="sqli"
MODEL_PRESET="qwen2.5-coder-1.5b"  # RTX 5090 32GB 可以跑 1.5B
PRETRAIN_EPOCHS=3
REWARD_SAMPLES=4000
RL_EPISODES=20

# RTX 5090 32GB 优化参数
BATCH_SIZE=32
GRADIENT_ACCUMULATION=4
MAX_LENGTH=256
DATALOADER_WORKERS=8

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ==================== 检查系统要求 ====================
print_header "检查系统要求"

# 检查是否为 Linux
if [[ "$(uname)" != "Linux" ]]; then
    print_error "此脚本仅支持 Linux 系统"
    exit 1
fi
print_success "操作系统: $(uname -a)"

# 检查 NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "未检测到 NVIDIA 驱动，请先安装 CUDA"
    exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
print_success "GPU: $GPU_INFO"

# 检查 Conda
if ! command -v conda &> /dev/null; then
    print_error "未检测到 Conda，请先安装 Anaconda 或 Miniconda"
    echo "安装命令: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
print_success "Conda: $(conda --version)"

# ==================== 创建 Conda 环境 ====================
print_header "创建 Conda 环境: $CONDA_ENV_NAME"

# 检查环境是否已存在
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    print_warning "环境 $CONDA_ENV_NAME 已存在"
    read -p "是否重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $CONDA_ENV_NAME -y
        conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    fi
else
    conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
fi

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME
print_success "已激活环境: $CONDA_ENV_NAME"

# ==================== 安装依赖 ====================
print_header "安装 Python 依赖"

# 升级 pip
pip install --upgrade pip

# 安装 PyTorch (CUDA 12.4 for RTX 5090)
print_success "安装 PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装编译依赖
print_success "安装基础工具..."
pip install psutil ninja packaging wheel setuptools

# 跳过 Flash Attention（编译复杂且不是必须的）
print_warning "跳过 Flash Attention 安装（RTX 5090 32GB 显存充足，不需要）"

# 安装训练依赖
print_success "安装训练依赖..."
pip install transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.28.0 \
    trl>=0.8.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    scipy \
    scikit-learn \
    pandas \
    tensorboard \
    wandb \
    tqdm \
    requests

# 安装项目依赖（如果有 requirements.txt）
if [ -f "train/requirements_train.txt" ]; then
    pip install -r train/requirements_train.txt
fi

print_success "依赖安装完成"

# 验证 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 不使用 Flash Attention（RTX 5090 32GB 不需要）
USE_FLASH_ATTN=""
print_success "使用默认注意力机制（显存充足）"

# ==================== 配置训练参数 (RTX 5090 32GB 优化) ====================
print_header "配置训练参数 (RTX 5090 32GB 优化)"

# 创建服务器优化配置
cat > train/config_server.py << 'EOF'
"""
服务器训练配置 - RTX 5090 32GB 优化
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List

# RTX 5090 32GB 优化预设
MODEL_PRESETS_SERVER = {
    "qwen2.5-coder-1.5b": {
        "model_name": "Qwen/Qwen2.5-Coder-1.5B",
        "max_length": 256,
        "batch_size": 32,  # RTX 5090 可以用更大 batch
        "gradient_accumulation": 4,  # 等效 batch=128
        "use_flash_attention": True,  # 启用 Flash Attention
    },
    "qwen2.5-coder-3b": {
        "model_name": "Qwen/Qwen2.5-Coder-3B",
        "max_length": 256,
        "batch_size": 16,
        "gradient_accumulation": 8,
        "use_flash_attention": True,
    },
    "qwen2.5-coder-7b": {
        "model_name": "Qwen/Qwen2.5-Coder-7B",
        "max_length": 256,
        "batch_size": 4,
        "gradient_accumulation": 32,
        "use_flash_attention": True,
        "load_in_4bit": True,  # 7B 需要量化
    },
}

@dataclass
class ServerTrainingConfig:
    """服务器训练配置"""
    output_dir: str = "models/pretrain_sqli_qwen"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    bf16: bool = True
    fp16: bool = False
    
    dataloader_num_workers: int = 8  # 服务器可以用更多 workers
    dataloader_pin_memory: bool = True
    
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    optim: str = "adamw_torch_fused"  # 使用 fused optimizer 加速
    
    seed: int = 42

EOF

print_success "服务器配置已创建"

# ==================== 设置 HuggingFace 镜像 (可选) ====================
print_header "配置 HuggingFace"

read -p "是否使用 HuggingFace 镜像加速下载? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
    print_success "已设置 HuggingFace 镜像: $HF_ENDPOINT"
fi

# ==================== 开始训练 ====================
print_header "开始训练流程"

echo "训练配置:"
echo "  攻击类型: $ATTACK_TYPE"
echo "  模型: $MODEL_PRESET"
echo "  预训练轮数: $PRETRAIN_EPOCHS"
echo "  奖励样本数: $REWARD_SAMPLES"
echo "  RL 轮数: $RL_EPISODES"
echo "  Batch Size: $BATCH_SIZE"
echo "  梯度累积: $GRADIENT_ACCUMULATION"
echo ""

read -p "确认开始训练? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_warning "训练已取消"
    exit 0
fi

# 设置路径
PRETRAIN_DIR="models/pretrain_${ATTACK_TYPE}_qwen2_5_coder_1_5b"
REWARD_DIR="models/reward_${ATTACK_TYPE}_qwen"
RL_DIR="models/rl_${ATTACK_TYPE}_qwen"
LABELED_DATA_DIR="data/labeled"

# -------------------- Stage 1: 预训练 --------------------
print_header "Stage 1: 预训练 ($MODEL_PRESET)"

if [ -d "$PRETRAIN_DIR" ]; then
    print_warning "预训练模型已存在: $PRETRAIN_DIR"
    read -p "跳过预训练? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_success "跳过预训练阶段"
    else
        python train/pretrain.py \
            --model-preset $MODEL_PRESET \
            --attack-type $ATTACK_TYPE \
            --output-dir $PRETRAIN_DIR \
            --epochs $PRETRAIN_EPOCHS \
            --batch-size $BATCH_SIZE \
            --gradient-accumulation $GRADIENT_ACCUMULATION \
            --bf16 $USE_FLASH_ATTN
        
        if [ $? -ne 0 ]; then
            print_error "预训练失败!"
            exit 1
        fi
    fi
else
    python train/pretrain.py \
        --model-preset $MODEL_PRESET \
        --attack-type $ATTACK_TYPE \
        --output-dir $PRETRAIN_DIR \
        --epochs $PRETRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation $GRADIENT_ACCUMULATION \
        --bf16 $USE_FLASH_ATTN
    
    if [ $? -ne 0 ]; then
        print_error "预训练失败!"
        exit 1
    fi
fi

print_success "预训练完成: $PRETRAIN_DIR"

# -------------------- Stage 2: 奖励模型训练 --------------------
print_header "Stage 2: 奖励模型训练"

REWARD_MODEL_PATH="$REWARD_DIR/final_reward_model"

if [ -d "$REWARD_MODEL_PATH" ]; then
    print_warning "奖励模型已存在: $REWARD_MODEL_PATH"
    read -p "跳过奖励模型训练? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_success "跳过奖励模型训练"
    else
        # 生成标注数据
        print_success "生成标注数据..."
        python train/generate_labeled_data.py \
            --attack_type $ATTACK_TYPE \
            --input_file "data/processed/$ATTACK_TYPE/train.txt" \
            --output_dir $LABELED_DATA_DIR \
            --num_samples $REWARD_SAMPLES \
            --waf_url "http://localhost:8081"
        
        # 训练奖励模型
        print_success "训练奖励模型..."
        python train/train_reward_model.py \
            --pretrained_model_path $PRETRAIN_DIR \
            --data_path $LABELED_DATA_DIR \
            --output_dir $REWARD_DIR \
            --batch_size 32 \
            --epochs 4 \
            --bf16
        
        if [ $? -ne 0 ]; then
            print_error "奖励模型训练失败!"
            exit 1
        fi
    fi
else
    # 检查标注数据
    TRAIN_CSV="$LABELED_DATA_DIR/${ATTACK_TYPE}_train.csv"
    if [ ! -f "$TRAIN_CSV" ]; then
        print_success "生成标注数据..."
        
        # 检查 WAF 是否运行
        if ! curl -s "http://localhost:8081" > /dev/null 2>&1; then
            print_warning "WAF 未运行在 localhost:8081"
            print_warning "请先启动 WAF 服务，或者使用已有的标注数据"
            read -p "继续? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        python train/generate_labeled_data.py \
            --attack_type $ATTACK_TYPE \
            --input_file "data/processed/$ATTACK_TYPE/train.txt" \
            --output_dir $LABELED_DATA_DIR \
            --num_samples $REWARD_SAMPLES \
            --waf_url "http://localhost:8081"
    fi
    
    # 训练奖励模型
    print_success "训练奖励模型..."
    python train/train_reward_model.py \
        --pretrained_model_path $PRETRAIN_DIR \
        --data_path $LABELED_DATA_DIR \
        --output_dir $REWARD_DIR \
        --batch_size 32 \
        --epochs 4 \
        --bf16
    
    if [ $? -ne 0 ]; then
        print_error "奖励模型训练失败!"
        exit 1
    fi
fi

print_success "奖励模型训练完成: $REWARD_MODEL_PATH"

# -------------------- Stage 3: 强化学习 (PPO) --------------------
print_header "Stage 3: 强化学习 (PPO)"

print_success "开始 RL 训练..."
python train/train_rl.py \
    --pretrained_model $PRETRAIN_DIR \
    --reward_model $REWARD_MODEL_PATH \
    --output_dir $RL_DIR \
    --batch_size 256 \
    --mini_batch_size 16 \
    --total_episodes $RL_EPISODES \
    --bf16

if [ $? -ne 0 ]; then
    print_error "RL 训练失败!"
    exit 1
fi

print_success "RL 训练完成: $RL_DIR"

# -------------------- 测试最终模型 --------------------
print_header "测试最终模型"

FINAL_MODEL_PATH="$RL_DIR/final_model"
if [ -d "$FINAL_MODEL_PATH" ]; then
    print_success "生成测试 payload..."
    python train/test_rl_model.py \
        --model_path $FINAL_MODEL_PATH \
        --num_samples 20 \
        --temperature 1.0
else
    print_warning "最终模型未找到: $FINAL_MODEL_PATH"
fi

# ==================== 完成 ====================
print_header "训练完成!"

echo ""
echo "输出模型:"
echo "  预训练: $PRETRAIN_DIR"
echo "  奖励模型: $REWARD_MODEL_PATH"
echo "  RL 模型: $RL_DIR/final_model"
echo ""
echo "后续操作:"
echo "  1. 测试: python train/test_rl_model.py --model_path $RL_DIR/final_model"
echo "  2. 生成: python train/generate_payloads.py --model_path $RL_DIR/final_model"
echo "  3. 评估: python train/evaluate_rl.py --model_path $RL_DIR/final_model"
echo ""

print_success "所有训练阶段完成!"
