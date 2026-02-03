# WAF-LLM训练环境配置脚本 (Windows PowerShell)
# 针对RTX 4070 8GB优化

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "WAF-LLM训练环境配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 检查conda环境
Write-Host "`n[1/5] 检查conda环境..." -ForegroundColor Yellow
$condaEnv = conda env list | Select-String "bas_fuzzer"
if ($condaEnv) {
    Write-Host "✅ 找到bas_fuzzer环境" -ForegroundColor Green
} else {
    Write-Host "❌ 未找到bas_fuzzer环境，请先创建" -ForegroundColor Red
    exit 1
}

# 激活环境
Write-Host "`n[2/5] 激活bas_fuzzer环境..." -ForegroundColor Yellow
conda activate bas_fuzzer

# 安装训练依赖
Write-Host "`n[3/5] 安装训练依赖..." -ForegroundColor Yellow
Write-Host "这可能需要几分钟时间..." -ForegroundColor Gray
pip install -r train/requirements_train.txt

# 验证PyTorch和CUDA
Write-Host "`n[4/5] 验证PyTorch和CUDA..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 验证GPU
Write-Host "`n[5/5] 测试GPU配置..." -ForegroundColor Yellow
python train/monitor_gpu.py

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "环境配置完成!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n下一步:" -ForegroundColor Yellow
Write-Host "1. 运行: python train/monitor_gpu.py (测试GPU)" -ForegroundColor White
Write-Host "2. 解压数据集: python train/prepare_data.py --extract" -ForegroundColor White
Write-Host "3. 开始预训练: python train/pretrain.py" -ForegroundColor White
