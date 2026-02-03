# ============================================================
# GPTFuzzer Full Training Pipeline - Qwen2.5-Coder-1.5B
# Optimized for RTX 4070 8GB
# ============================================================

param(
    [string]$AttackType = "sqli",
    [string]$ModelPreset = "qwen2.5-coder-1.5b",
    [int]$PretrainEpochs = 3,
    [int]$RewardSamples = 4000,
    [int]$RLEpisodes = 20,
    [switch]$SkipDeps = $false
)

$ErrorActionPreference = "Stop"

Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "GPTFuzzer Training Pipeline - Qwen2.5-Coder" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

# ============================================================
# Check and Install Dependencies
# ============================================================
if (-not $SkipDeps) {
    Write-Host ""
    Write-Host "Checking dependencies..." -ForegroundColor Yellow
    
    $reqFile = "train/requirements_train.txt"
    if (Test-Path $reqFile) {
        Write-Host "Installing dependencies from $reqFile ..." -ForegroundColor Cyan
        pip install -r $reqFile -q
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Some dependencies may have failed to install" -ForegroundColor Yellow
        }
        else {
            Write-Host "Dependencies installed successfully" -ForegroundColor Green
        }
    }
    else {
        Write-Host "Requirements file not found: $reqFile" -ForegroundColor Yellow
    }
}
else {
    Write-Host "Skipping dependency check (--SkipDeps)" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Attack Type: $AttackType"
Write-Host "  Model Preset: $ModelPreset"
Write-Host "  Pretrain Epochs: $PretrainEpochs"
Write-Host "  Reward Samples: $RewardSamples"
Write-Host "  RL Episodes: $RLEpisodes"
Write-Host ""

# Set paths
$PretrainModelName = $ModelPreset -replace "\.", "_" -replace "-", "_"
$PretrainDir = "models/pretrain_${AttackType}_${PretrainModelName}"
$RewardDir = "models/reward_${AttackType}_qwen"
$RLDir = "models/rl_${AttackType}_qwen"

# ============================================================
# Stage 1: Pretraining
# ============================================================
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "Stage 1: Pretraining ($ModelPreset)" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

if (Test-Path $PretrainDir) {
    Write-Host "Pretrained model exists: $PretrainDir" -ForegroundColor Yellow
    $response = Read-Host "Skip pretraining? (Y/n)"
    if ($response -ne "n" -and $response -ne "N") {
        Write-Host "Skipping pretraining stage" -ForegroundColor Yellow
    }
    else {
        Write-Host "Re-training pretrained model..." -ForegroundColor Cyan
        python train/pretrain.py --model-preset $ModelPreset --attack-type $AttackType --output-dir $PretrainDir --epochs $PretrainEpochs --bf16
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Pretraining failed!" -ForegroundColor Red
            exit 1
        }
    }
}
else {
    Write-Host "Starting pretraining..." -ForegroundColor Cyan
    python train/pretrain.py --model-preset $ModelPreset --attack-type $AttackType --output-dir $PretrainDir --epochs $PretrainEpochs --bf16
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pretraining failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Pretraining completed: $PretrainDir" -ForegroundColor Green

# ============================================================
# Stage 2: Reward Model Training
# ============================================================
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "Stage 2: Reward Model Training" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

$LabeledDataDir = "data/labeled"
$RewardModelPath = "$RewardDir/final_reward_model"

if (Test-Path $RewardModelPath) {
    Write-Host "Reward model exists: $RewardModelPath" -ForegroundColor Yellow
    $response = Read-Host "Skip reward model training? (Y/n)"
    if ($response -ne "n" -and $response -ne "N") {
        Write-Host "Skipping reward model training" -ForegroundColor Yellow
    }
    else {
        # Generate labeled data
        Write-Host "Generating labeled data..." -ForegroundColor Cyan
        python train/generate_labeled_data.py --attack_type $AttackType --input_file "data/processed/$AttackType/train.txt" --output_dir $LabeledDataDir --num_samples $RewardSamples --waf_url "http://localhost:8081"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Labeled data generation failed!" -ForegroundColor Red
            exit 1
        }
        
        # Train reward model
        Write-Host "Training reward model..." -ForegroundColor Cyan
        python train/train_reward_model.py --pretrained_model_path $PretrainDir --data_path $LabeledDataDir --output_dir $RewardDir --batch_size 16 --epochs 4 --bf16
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Reward model training failed!" -ForegroundColor Red
            exit 1
        }
    }
}
else {
    # Check if labeled data exists
    $TrainCSV = "$LabeledDataDir/${AttackType}_train.csv"
    if (-not (Test-Path $TrainCSV)) {
        Write-Host "Generating labeled data..." -ForegroundColor Cyan
        python train/generate_labeled_data.py --attack_type $AttackType --input_file "data/processed/$AttackType/train.txt" --output_dir $LabeledDataDir --num_samples $RewardSamples --waf_url "http://localhost:8081"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Labeled data generation failed!" -ForegroundColor Red
            exit 1
        }
    }
    
    # Train reward model
    Write-Host "Training reward model..." -ForegroundColor Cyan
    python train/train_reward_model.py --pretrained_model_path $PretrainDir --data_path $LabeledDataDir --output_dir $RewardDir --batch_size 16 --epochs 4 --bf16
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Reward model training failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Reward model training completed: $RewardModelPath" -ForegroundColor Green

# ============================================================
# Stage 3: Reinforcement Learning (PPO)
# ============================================================
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "Stage 3: Reinforcement Learning (PPO)" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

Write-Host "Starting RL training..." -ForegroundColor Cyan
python train/train_rl.py --pretrained_model $PretrainDir --reward_model $RewardModelPath --output_dir $RLDir --batch_size 128 --mini_batch_size 8 --total_episodes $RLEpisodes --bf16

if ($LASTEXITCODE -ne 0) {
    Write-Host "RL training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "RL training completed: $RLDir" -ForegroundColor Green

# ============================================================
# Test Final Model
# ============================================================
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "Testing Final Model" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

$FinalModelPath = "$RLDir/final_model"
if (Test-Path $FinalModelPath) {
    Write-Host "Generating test payloads..." -ForegroundColor Cyan
    python train/test_rl_model.py --model_path $FinalModelPath --num_samples 20 --temperature 1.0
}
else {
    Write-Host "Final model not found: $FinalModelPath" -ForegroundColor Red
}

# ============================================================
# Done
# ============================================================
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "Output Models:" -ForegroundColor Yellow
Write-Host "  Pretrained: $PretrainDir"
Write-Host "  Reward: $RewardModelPath"
Write-Host "  RL: $RLDir/final_model"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Test: python train/test_rl_model.py --model_path $RLDir/final_model"
Write-Host "  2. Generate: python train/generate_payloads.py --model_path $RLDir/final_model"
Write-Host "  3. Evaluate: python train/evaluate_rl.py --model_path $RLDir/final_model"
