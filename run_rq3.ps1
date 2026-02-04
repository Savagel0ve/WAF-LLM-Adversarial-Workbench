# RQ3: 预训练数据规模影响实验
# =============================
# 测试不同预训练数据规模对效果的影响

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [string[]]$AttackTypes = @("sqli", "xss"),
    [int[]]$DataScales = @(0, 20000, 256000, 512000),
    [int]$Seed = 42
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RQ3: 预训练数据规模影响实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:"
Write-Host "  输出目录: $OutputDir"
Write-Host "  攻击类型: $($AttackTypes -join ', ')"
Write-Host "  数据规模: $($DataScales -join ', ')"
Write-Host ""

$CondaEnv = "waf-llm"
Write-Host "激活conda环境: $CondaEnv" -ForegroundColor Yellow

$AttackTypesStr = $AttackTypes -join " "
$DataScalesStr = $DataScales -join " "

$Command = @"
python experiments/rq3_data_scale.py `
    --output_dir "$OutputDir" `
    --data_dir "$DataDir" `
    --models_dir "$ModelsDir" `
    --attack_types $AttackTypesStr `
    --data_scales $DataScalesStr `
    --seed $Seed
"@

Write-Host "开始运行实验..." -ForegroundColor Green
Write-Host $Command -ForegroundColor Gray
Write-Host ""

Invoke-Expression "conda activate $CondaEnv; $Command"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "RQ3实验完成! 结果保存在: $OutputDir/rq3_data_scale" -ForegroundColor Green
} else {
    Write-Host "实验失败" -ForegroundColor Red
}
