# RQ5: 超参数影响实验
# ====================
# 测试KL系数和奖励模型数据量的影响

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [string[]]$AttackTypes = @("sqli", "xss", "rce"),
    [double[]]$KLCoefficients = @(0, 0.1, 0.2, 0.5, 1.0),
    [int[]]$RewardDataSizes = @(2000, 4000),
    [int]$Seed = 42
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RQ5: 超参数影响实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:"
Write-Host "  输出目录: $OutputDir"
Write-Host "  攻击类型: $($AttackTypes -join ', ')"
Write-Host "  KL系数: $($KLCoefficients -join ', ')"
Write-Host "  奖励数据量: $($RewardDataSizes -join ', ')"
Write-Host ""

$CondaEnv = "waf-llm"
Write-Host "激活conda环境: $CondaEnv" -ForegroundColor Yellow

$AttackTypesStr = $AttackTypes -join " "
$KLStr = $KLCoefficients -join " "
$DataSizesStr = $RewardDataSizes -join " "

$Command = @"
python experiments/rq5_hyperparams.py `
    --output_dir "$OutputDir" `
    --data_dir "$DataDir" `
    --models_dir "$ModelsDir" `
    --attack_types $AttackTypesStr `
    --kl_coefficients $KLStr `
    --reward_data_sizes $DataSizesStr `
    --seed $Seed
"@

Write-Host "开始运行实验..." -ForegroundColor Green
Write-Host $Command -ForegroundColor Gray
Write-Host ""

Invoke-Expression "conda activate $CondaEnv; $Command"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "RQ5实验完成! 结果保存在: $OutputDir/rq5_hyperparams" -ForegroundColor Green
} else {
    Write-Host "实验失败" -ForegroundColor Red
}
