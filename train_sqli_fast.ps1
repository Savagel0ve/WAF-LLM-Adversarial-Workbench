# SQLi Pretraining Script - Optimized for Speed
# RTX 4070 8GB - Fast Training Configuration

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPTFuzzer SQLi Fast Training" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

# Force using CUDA device 0 (RTX 4070)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_LAUNCH_BLOCKING = "0"  # Set to 0 for better performance

# Check GPU
Write-Host "`nChecking GPU..." -ForegroundColor Yellow
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "`nOptimizations enabled:" -ForegroundColor Yellow
Write-Host "  - Larger batch size (8 -> 2x faster)" -ForegroundColor Green
Write-Host "  - Gradient checkpointing OFF (-> 2x faster)" -ForegroundColor Green
Write-Host "  - Shorter sequence length (96 -> 1.5x faster)" -ForegroundColor Green
Write-Host "  - Expected speedup: ~6x faster!" -ForegroundColor Green

Write-Host "`nStarting optimized training..." -ForegroundColor Green
Write-Host "Log file: logs\pretrain_sqli_fast.log`n" -ForegroundColor Gray

# Start training with optimized parameters
python train/pretrain.py `
    --attack-type sqli `
    --model-name gpt2 `
    --epochs 3 `
    --batch-size 8 `
    --gradient-accumulation 4 `
    --fp16 `
    --learning-rate 5e-5 `
    --max-length 96 `
    --logging-steps 50 `
    --save-steps 500 `
    --eval-steps 500 `
    2>&1 | Tee-Object -FilePath "logs\pretrain_sqli_fast.log"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
