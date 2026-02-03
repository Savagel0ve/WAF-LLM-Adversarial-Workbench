# GPTFuzzer Stage 3: Reinforcement Learning Training Script (SQLi)
# Use PPO algorithm to fine-tune pretrained model for WAF bypass

# Color output function
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green @"
================================================================
    GPTFuzzer Stage 3: Reinforcement Learning
    Training WAF Bypass Model with PPO
================================================================
"@

# Activate virtual environment
Write-ColorOutput Cyan "`n[1/5] Activating virtual environment..."
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
    Write-ColorOutput Green "Virtual environment activated"
} else {
    Write-ColorOutput Yellow "Virtual environment not found, using global Python"
}

# Check required models
Write-ColorOutput Cyan "`n[2/5] Checking required models..."

$pretrainModel = ".\models\pretrain_sqli_gpt2_small"
$rewardModel = ".\models\reward_sqli\final_reward_model"

$allModelsExist = $true

if (Test-Path $pretrainModel) {
    Write-ColorOutput Green "Pretrained model found: $pretrainModel"
} else {
    Write-ColorOutput Red "Pretrained model not found: $pretrainModel"
    Write-ColorOutput Yellow "  Please run train_sqli.ps1 first to complete pretraining"
    $allModelsExist = $false
}

if (Test-Path $rewardModel) {
    Write-ColorOutput Green "Reward model found: $rewardModel"
} else {
    Write-ColorOutput Red "Reward model not found: $rewardModel"
    Write-ColorOutput Yellow "  Please run train_reward_sqli.ps1 first to train the reward model"
    $allModelsExist = $false
}

if (-not $allModelsExist) {
    Write-ColorOutput Red "`nError: Required models missing, cannot continue"
    exit 1
}

# Check CUDA
Write-ColorOutput Cyan "`n[3/5] Checking CUDA environment..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Yellow "Cannot detect CUDA, will use CPU training (very slow)"
} else {
    Write-ColorOutput Green "CUDA environment OK"
}

# Configure parameters
Write-ColorOutput Cyan "`n[4/5] Configuring training parameters..."

$OUTPUT_DIR = ".\models\rl_sqli_gpt2"
$PRETRAIN_MODEL = $pretrainModel
$REWARD_MODEL = $rewardModel

# Training parameters (from paper)
$LEARNING_RATE = "1.4e-5"  # Paper recommended value
$BATCH_SIZE = 256           # Paper recommended value
$MINI_BATCH_SIZE = 16       # Memory optimization: 256/16 = 16 accumulations
$KL_COEF = 0.2              # Beta parameter, critical!
$TOTAL_EPISODES = 20        # Total training episodes
$MAX_NEW_TOKENS = 128       # Max generation length
$SAVE_FREQ = 5              # Save every 5 episodes

Write-ColorOutput Yellow @"
Training Configuration:
  - Pretrained model: $PRETRAIN_MODEL
  - Reward model: $REWARD_MODEL
  - Output directory: $OUTPUT_DIR
  - Learning rate: $LEARNING_RATE (paper recommended)
  - Batch size: $BATCH_SIZE (paper recommended)
  - Mini batch size: $MINI_BATCH_SIZE
  - KL coefficient: $KL_COEF (paper recommended)
  - Total episodes: $TOTAL_EPISODES
  - Max new tokens: $MAX_NEW_TOKENS
"@

# Start training
Write-ColorOutput Green "`n[5/5] Starting reinforcement learning training..."
Write-ColorOutput Cyan "="*70

$trainingStartTime = Get-Date

# Execute training
python train/train_rl.py `
    --pretrained_model $PRETRAIN_MODEL `
    --reward_model $REWARD_MODEL `
    --output_dir $OUTPUT_DIR `
    --learning_rate $LEARNING_RATE `
    --batch_size $BATCH_SIZE `
    --mini_batch_size $MINI_BATCH_SIZE `
    --init_kl_coef $KL_COEF `
    --total_episodes $TOTAL_EPISODES `
    --max_new_tokens $MAX_NEW_TOKENS `
    --save_freq $SAVE_FREQ `
    --seed 42

$exitCode = $LASTEXITCODE
$trainingEndTime = Get-Date
$trainingDuration = $trainingEndTime - $trainingStartTime

Write-ColorOutput Cyan "`n"
Write-ColorOutput Cyan "="*70

if ($exitCode -eq 0) {
    Write-ColorOutput Green @"

================================================================
    Training Completed Successfully!
================================================================

Training Statistics:
  - Training time: $($trainingDuration.ToString("hh\:mm\:ss"))
  - Model saved at: $OUTPUT_DIR
  - Final model: $OUTPUT_DIR\final_model

Next Steps:
  1. Test generation quality:
     python train/test_rl_model.py --model_path $OUTPUT_DIR\final_model

  2. Evaluate WAF bypass rate:
     python train/evaluate_rl.py --model_path $OUTPUT_DIR\final_model

  3. Generate payloads in bulk:
     python train/generate_payloads.py --model_path $OUTPUT_DIR\final_model --num_samples 1000

"@
} else {
    Write-ColorOutput Red @"

================================================================
    Training Failed
================================================================

Possible causes:
  1. Out of memory (OOM)
     Solution: Reduce batch_size or mini_batch_size

  2. Model loading failed
     Solution: Check if model paths are correct

  3. Dependency issues
     Solution: Run pip install -r train/requirements_train.txt

Please check the error message above and fix the issue before re-running.

"@
    exit $exitCode
}

# Show memory usage
Write-ColorOutput Cyan "`nGPU Memory Usage:"
python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB'); print(f'Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB')" 2>$null

Write-ColorOutput Green "`nTraining script completed!"
