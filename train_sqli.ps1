# SQLi Pretraining Script - Force using RTX 4070
# Ensure using dedicated GPU instead of integrated GPU

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPTFuzzer SQLi Pretraining" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

# Force using CUDA device 0 (RTX 4070)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_LAUNCH_BLOCKING = "1"

# Check GPU
Write-Host "`nChecking GPU..." -ForegroundColor Yellow
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "`nStarting training..." -ForegroundColor Green
Write-Host "Log file: logs\pretrain_sqli.log`n" -ForegroundColor Gray

# Start training
python train/pretrain.py `
    --attack-type sqli `
    --model-name gpt2 `
    --epochs 5 `
    --batch-size 4 `
    --gradient-accumulation 8 `
    --fp16 `
    --learning-rate 5e-5 `
    --max-length 128 `
    --logging-steps 50 `
    --save-steps 1000 `
    --eval-steps 1000 `
    2>&1 | Tee-Object -FilePath "logs\pretrain_sqli.log"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
