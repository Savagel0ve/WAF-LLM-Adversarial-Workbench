# RQ6: 超越语法能力实验
# =======================
# 测试GPTFuzzer生成语法外新payload的能力

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [string[]]$AttackTypes = @("sqli"),
    [int]$NumGenerate = 500000,
    [int]$Seed = 42
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RQ6: 超越语法能力实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:"
Write-Host "  输出目录: $OutputDir"
Write-Host "  攻击类型: $($AttackTypes -join ', ')"
Write-Host "  生成数量: $NumGenerate"
Write-Host ""
Write-Host "注意: 此实验需要生成大量payload，可能需要较长时间" -ForegroundColor Yellow
Write-Host ""

$CondaEnv = "waf-llm"
Write-Host "激活conda环境: $CondaEnv" -ForegroundColor Yellow

$AttackTypesStr = $AttackTypes -join " "

$Command = @"
python experiments/rq6_novel_payloads.py `
    --output_dir "$OutputDir" `
    --data_dir "$DataDir" `
    --models_dir "$ModelsDir" `
    --attack_types $AttackTypesStr `
    --num_generate $NumGenerate `
    --seed $Seed
"@

Write-Host "开始运行实验..." -ForegroundColor Green
Write-Host $Command -ForegroundColor Gray
Write-Host ""

Invoke-Expression "conda activate $CondaEnv; $Command"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "RQ6实验完成! 结果保存在: $OutputDir/rq6_novel_payloads" -ForegroundColor Green
} else {
    Write-Host "实验失败" -ForegroundColor Red
}
