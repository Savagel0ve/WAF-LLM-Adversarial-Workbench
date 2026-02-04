# RQ4: 奖励模型 vs WAF反馈实验
# =============================
# 对比奖励模型指导RL与直接WAF反馈的效果

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [string[]]$AttackTypes = @("sqli", "xss", "rce"),
    [int]$RLEpochs = 20,
    [int]$RLBatchSize = 256,
    [int]$Seed = 42
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RQ4: 奖励模型 vs WAF反馈实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:"
Write-Host "  输出目录: $OutputDir"
Write-Host "  攻击类型: $($AttackTypes -join ', ')"
Write-Host "  RL轮数: $RLEpochs"
Write-Host "  RL批次: $RLBatchSize"
Write-Host ""

$CondaEnv = "waf-llm"
Write-Host "激活conda环境: $CondaEnv" -ForegroundColor Yellow

$AttackTypesStr = $AttackTypes -join " "

$Command = @"
python experiments/rq4_reward_vs_waf.py `
    --output_dir "$OutputDir" `
    --data_dir "$DataDir" `
    --models_dir "$ModelsDir" `
    --attack_types $AttackTypesStr `
    --rl_epochs $RLEpochs `
    --rl_batch_size $RLBatchSize `
    --seed $Seed
"@

Write-Host "开始运行实验..." -ForegroundColor Green
Write-Host $Command -ForegroundColor Gray
Write-Host ""

Invoke-Expression "conda activate $CondaEnv; $Command"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "RQ4实验完成! 结果保存在: $OutputDir/rq4_reward_vs_waf" -ForegroundColor Green
} else {
    Write-Host "实验失败" -ForegroundColor Red
}
