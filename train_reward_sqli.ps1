# SQLi Reward Model Training Pipeline
# Step 1: Generate labeled data
# Step 2: Train reward model

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SQLi Reward Model Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Configuration
$ATTACK_TYPE = "sqli"
$NUM_SAMPLES = 4000  # Paper: SQLi uses 4000 samples
$WAF_URL = "http://localhost:8081"
$PRETRAINED_MODEL = ".\models\pretrain_sqli_gpt2_small"
$OUTPUT_DIR = ".\models\reward_sqli"
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
    Write-Host "Please ensure WAF service is running!" -ForegroundColor Yellow
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
    Write-Host "Skipping data labeling step. Use --force to regenerate." -ForegroundColor Yellow
} else {
    $INPUT_FILE = ".\data\processed\sqli\train.txt"

    if (-not (Test-Path $INPUT_FILE)) {
        Write-Host "ERROR: Input file not found: $INPUT_FILE" -ForegroundColor Red
        exit 1
    }

    Write-Host "Input file: $INPUT_FILE" -ForegroundColor White
    Write-Host "Sample count: $NUM_SAMPLES" -ForegroundColor White
    Write-Host "WAF URL: $WAF_URL" -ForegroundColor White

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

    Write-Host "Data labeling completed!" -ForegroundColor Green
}

# Step 2: Train reward model
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 2: Train Reward Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path $LABELED_DATA_TRAIN)) {
    Write-Host "ERROR: Labeled data not found: $LABELED_DATA_TRAIN" -ForegroundColor Red
    exit 1
}

Write-Host "Pretrained model: $PRETRAINED_MODEL" -ForegroundColor White
Write-Host "Labeled data: $DATA_DIR" -ForegroundColor White
Write-Host "Output directory: $OUTPUT_DIR" -ForegroundColor White

# Training parameters (paper defaults)
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

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nModel saved to: $OUTPUT_DIR\final_reward_model" -ForegroundColor Green

# Step 3: Test model
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 3: Test Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$TEST_FILE = ".\data\processed\sqli\test.txt"

if (Test-Path $TEST_FILE) {
    Write-Host "Using test file: $TEST_FILE" -ForegroundColor White
    
    python train\test_reward_model.py `
        --model_path "$OUTPUT_DIR\final_reward_model" `
        --payload_file $TEST_FILE `
        --batch_size 32
} else {
    Write-Host "Test file not found, skipping test" -ForegroundColor Yellow
}

Write-Host "`nView training logs:" -ForegroundColor Cyan
Write-Host "tensorboard --logdir $OUTPUT_DIR\logs" -ForegroundColor White
