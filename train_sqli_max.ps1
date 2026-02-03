# SQLi Pretraining Script - Maximum Performance
# RTX 4070 8GB - Aggressive optimization to use ~80% VRAM

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GPTFuzzer SQLi MAX SPEED Training" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan

# Force using CUDA device 0 (RTX 4070)
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Check GPU
Write-Host "`nChecking GPU..." -ForegroundColor Yellow
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "`nAggressive Optimizations:" -ForegroundColor Yellow
Write-Host "  - Batch size 16 (4x original)" -ForegroundColor Green
Write-Host "  - Gradient accumulation 2" -ForegroundColor Green
Write-Host "  - Effective batch size 32" -ForegroundColor Green
Write-Host "  - Gradient checkpointing OFF" -ForegroundColor Green
Write-Host "  - Target VRAM usage: ~80% (6.4GB)" -ForegroundColor Green
Write-Host "  - Expected speedup: ~10x faster!" -ForegroundColor Magenta

Write-Host "`nStarting MAX SPEED training..." -ForegroundColor Green
Write-Host "Log file: logs\pretrain_sqli_max.log`n" -ForegroundColor Gray

# Start training with aggressive parameters
python train/pretrain.py `
    --attack-type sqli `
    --model-name gpt2 `
    --epochs 3 `
    --batch-size 16 `
    --gradient-accumulation 2 `
    --fp16 `
    --learning-rate 5e-5 `
    --max-length 128 `
    --logging-steps 50 `
    --save-steps 500 `
    --eval-steps 500 `
    2>&1 | Tee-Object -FilePath "logs\pretrain_sqli_max.log"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
