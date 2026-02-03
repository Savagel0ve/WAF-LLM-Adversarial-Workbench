# XSS Reward Model Training Pipeline

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "XSS Reward Model Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Configuration
$ATTACK_TYPE = "xss"
$NUM_SAMPLES = 2000  # Paper: XSS uses 2000 samples
$WAF_URL = "http://localhost:8081"
$PRETRAINED_MODEL = ".\models\pretrain_xss_gpt2_small"
$OUTPUT_DIR = ".\models\reward_xss"
$DATA_DIR = ".\data\labeled"

# Check pretrained model
if (-not (Test-Path $PRETRAINED_MODEL)) {
    Write-Host "ERROR: Pretrained model not found: $PRETRAINED_MODEL" -ForegroundColor Red
    Write-Host "Please complete pretraining first!" -ForegroundColor Yellow
    exit 1
}

# Check WAF connection
Write-Host "`nChecking WAF connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri $WAF_URL -TimeoutSec 5 -UseBasicParsing
    Write-Host "WAF connection successful: $WAF_URL" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Cannot connect to WAF: $WAF_URL" -ForegroundColor Red
    exit 1
}

# Step 1: Generate labeled data
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 1: Generate Labeled Data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$LABELED_DATA_TRAIN = "$DATA_DIR\${ATTACK_TYPE}_train.csv"
$LABELED_DATA_VAL = "$DATA_DIR\${ATTACK_TYPE}_val.csv"
$LABELED_DATA_TEST = "$DATA_DIR\${ATTACK_TYPE}_test.csv"

# Check if labeled data already exists
if ((Test-Path $LABELED_DATA_TRAIN) -and (Test-Path $LABELED_DATA_VAL) -and (Test-Path $LABELED_DATA_TEST)) {
    Write-Host "Labeled data already exists:" -ForegroundColor Green
    Write-Host "  - $LABELED_DATA_TRAIN" -ForegroundColor White
    Write-Host "  - $LABELED_DATA_VAL" -ForegroundColor White
    Write-Host "  - $LABELED_DATA_TEST" -ForegroundColor White
    Write-Host "Skipping data labeling step. Delete files to regenerate." -ForegroundColor Yellow
} else {
    $INPUT_FILE = ".\data\processed\xss\train.txt"

    if (-not (Test-Path $INPUT_FILE)) {
        Write-Host "ERROR: Input file not found: $INPUT_FILE" -ForegroundColor Red
        exit 1
    }

    python train\generate_labeled_data.py `
        --attack_type $ATTACK_TYPE `
        --input_file $INPUT_FILE `
        --output_dir $DATA_DIR `
        --num_samples $NUM_SAMPLES `
        --waf_url $WAF_URL `
        --balance_ratio 0.5 `
        --rate_limit 0.1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Data labeling failed!" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Train reward model
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 2: Train Reward Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python train\train_reward_model.py `
    --pretrained_model_path $PRETRAINED_MODEL `
    --data_path $DATA_DIR `
    --output_dir $OUTPUT_DIR `
    --max_length 128 `
    --batch_size 32 `
    --learning_rate 2e-5 `
    --epochs 4 `
    --warmup_ratio 0.1 `
    --weight_decay 0.01 `
    --fp16 `
    --early_stopping_patience 2

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Model training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nTraining completed!" -ForegroundColor Green
Write-Host "Model saved to: $OUTPUT_DIR\final_reward_model" -ForegroundColor Green

# Step 3: Test model
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 3: Test Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$TEST_FILE = ".\data\processed\xss\test.txt"

if (Test-Path $TEST_FILE) {
    python train\test_reward_model.py `
        --model_path "$OUTPUT_DIR\final_reward_model" `
        --payload_file $TEST_FILE `
        --batch_size 32
}

Write-Host "`nView training logs:" -ForegroundColor Cyan
Write-Host "tensorboard --logdir $OUTPUT_DIR\logs" -ForegroundColor White
