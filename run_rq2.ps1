# RQ2: 攻击语法影响实验
# =======================
# 对比短序列和长序列的效果差异

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [string[]]$AttackTypes = @("sqli", "xss", "rce"),
    [int]$NumPayloads = 100000,
    [int]$Seed = 42
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RQ2: 攻击语法影响实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:"
Write-Host "  输出目录: $OutputDir"
Write-Host "  攻击类型: $($AttackTypes -join ', ')"
Write-Host "  Payload数量: $NumPayloads"
Write-Host ""

$CondaEnv = "waf-llm"
Write-Host "激活conda环境: $CondaEnv" -ForegroundColor Yellow

$AttackTypesStr = $AttackTypes -join " "

$Command = @"
python experiments/rq2_grammar.py `
    --output_dir "$OutputDir" `
    --data_dir "$DataDir" `
    --models_dir "$ModelsDir" `
    --attack_types $AttackTypesStr `
    --num_payloads $NumPayloads `
    --seed $Seed
"@

Write-Host "开始运行实验..." -ForegroundColor Green
Write-Host $Command -ForegroundColor Gray
Write-Host ""

Invoke-Expression "conda activate $CondaEnv; $Command"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "RQ2实验完成! 结果保存在: $OutputDir/rq2_grammar" -ForegroundColor Green
} else {
    Write-Host "实验失败" -ForegroundColor Red
}
