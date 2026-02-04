# RQ1: GPTFuzzer有效性与效率实验
# ================================
# 对比GPTFuzzer与基线方法的TP和效率

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [string[]]$AttackTypes = @("sqli"),
    [string[]]$WafTypes = @("modsecurity"),
    [string[]]$Methods = @("gptfuzzer", "random_fuzzer", "grammar_rl"),
    [int]$BatchSize = 100,
    [int]$Seed = 42
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "RQ1: GPTFuzzer有效性与效率实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:"
Write-Host "  输出目录: $OutputDir"
Write-Host "  攻击类型: $($AttackTypes -join ', ')"
Write-Host "  WAF类型: $($WafTypes -join ', ')"
Write-Host "  对比方法: $($Methods -join ', ')"
Write-Host "  批次大小: $BatchSize"
Write-Host ""

# 激活conda环境
$CondaEnv = "waf-llm"
Write-Host "激活conda环境: $CondaEnv" -ForegroundColor Yellow

# 检查conda环境
$CondaPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $CondaPath) {
    Write-Host "错误: 未找到conda，请先安装Miniconda/Anaconda" -ForegroundColor Red
    exit 1
}

# 运行实验
Write-Host ""
Write-Host "开始运行实验..." -ForegroundColor Green

$AttackTypesStr = $AttackTypes -join " "
$WafTypesStr = $WafTypes -join " "
$MethodsStr = $Methods -join " "

$Command = @"
python experiments/rq1_effectiveness.py `
    --output_dir "$OutputDir" `
    --data_dir "$DataDir" `
    --models_dir "$ModelsDir" `
    --attack_types $AttackTypesStr `
    --waf_types $WafTypesStr `
    --methods $MethodsStr `
    --batch_size $BatchSize `
    --seed $Seed
"@

Write-Host "执行命令:" -ForegroundColor Gray
Write-Host $Command -ForegroundColor Gray
Write-Host ""

# 执行
Invoke-Expression "conda activate $CondaEnv; $Command"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "RQ1实验完成!" -ForegroundColor Green
    Write-Host "结果保存在: $OutputDir/rq1_effectiveness" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "实验失败，退出码: $LASTEXITCODE" -ForegroundColor Red
}
