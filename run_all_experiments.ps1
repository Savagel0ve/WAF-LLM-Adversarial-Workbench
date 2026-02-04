# 运行所有GPTFuzzer实验
# ======================
# 按顺序运行RQ1-RQ6所有实验

param(
    [string]$OutputDir = "results",
    [string]$DataDir = "data",
    [string]$ModelsDir = "models",
    [switch]$SkipRQ1,
    [switch]$SkipRQ2,
    [switch]$SkipRQ3,
    [switch]$SkipRQ4,
    [switch]$SkipRQ5,
    [switch]$SkipRQ6,
    [switch]$GenerateFigures,
    [int]$Seed = 42
)

$StartTime = Get-Date

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       GPTFuzzer 论文复现实验 - 完整运行" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "开始时间: $StartTime"
Write-Host "输出目录: $OutputDir"
Write-Host ""

# 创建输出目录
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# 记录日志
$LogFile = Join-Path $OutputDir "experiment_log.txt"
"GPTFuzzer实验日志" | Out-File $LogFile
"开始时间: $StartTime" | Out-File $LogFile -Append
"" | Out-File $LogFile -Append

# 运行各实验
$ExperimentResults = @{}

# RQ1
if (-not $SkipRQ1) {
    Write-Host ""
    Write-Host "========== RQ1: 有效性与效率 ==========" -ForegroundColor Yellow
    $RQ1Start = Get-Date
    
    & "$PSScriptRoot\run_rq1.ps1" -OutputDir $OutputDir -DataDir $DataDir -ModelsDir $ModelsDir -Seed $Seed
    
    $RQ1End = Get-Date
    $RQ1Duration = $RQ1End - $RQ1Start
    $ExperimentResults["RQ1"] = @{
        "Status" = if ($LASTEXITCODE -eq 0) { "Success" } else { "Failed" }
        "Duration" = $RQ1Duration.ToString()
    }
    "RQ1: $($ExperimentResults['RQ1'].Status), 耗时: $($RQ1Duration.ToString())" | Out-File $LogFile -Append
}

# RQ2
if (-not $SkipRQ2) {
    Write-Host ""
    Write-Host "========== RQ2: 攻击语法影响 ==========" -ForegroundColor Yellow
    $RQ2Start = Get-Date
    
    & "$PSScriptRoot\run_rq2.ps1" -OutputDir $OutputDir -DataDir $DataDir -ModelsDir $ModelsDir -Seed $Seed
    
    $RQ2End = Get-Date
    $RQ2Duration = $RQ2End - $RQ2Start
    $ExperimentResults["RQ2"] = @{
        "Status" = if ($LASTEXITCODE -eq 0) { "Success" } else { "Failed" }
        "Duration" = $RQ2Duration.ToString()
    }
    "RQ2: $($ExperimentResults['RQ2'].Status), 耗时: $($RQ2Duration.ToString())" | Out-File $LogFile -Append
}

# RQ3
if (-not $SkipRQ3) {
    Write-Host ""
    Write-Host "========== RQ3: 预训练数据规模 ==========" -ForegroundColor Yellow
    $RQ3Start = Get-Date
    
    & "$PSScriptRoot\run_rq3.ps1" -OutputDir $OutputDir -DataDir $DataDir -ModelsDir $ModelsDir -Seed $Seed
    
    $RQ3End = Get-Date
    $RQ3Duration = $RQ3End - $RQ3Start
    $ExperimentResults["RQ3"] = @{
        "Status" = if ($LASTEXITCODE -eq 0) { "Success" } else { "Failed" }
        "Duration" = $RQ3Duration.ToString()
    }
    "RQ3: $($ExperimentResults['RQ3'].Status), 耗时: $($RQ3Duration.ToString())" | Out-File $LogFile -Append
}

# RQ4
if (-not $SkipRQ4) {
    Write-Host ""
    Write-Host "========== RQ4: 奖励模型vs WAF ==========" -ForegroundColor Yellow
    $RQ4Start = Get-Date
    
    & "$PSScriptRoot\run_rq4.ps1" -OutputDir $OutputDir -DataDir $DataDir -ModelsDir $ModelsDir -Seed $Seed
    
    $RQ4End = Get-Date
    $RQ4Duration = $RQ4End - $RQ4Start
    $ExperimentResults["RQ4"] = @{
        "Status" = if ($LASTEXITCODE -eq 0) { "Success" } else { "Failed" }
        "Duration" = $RQ4Duration.ToString()
    }
    "RQ4: $($ExperimentResults['RQ4'].Status), 耗时: $($RQ4Duration.ToString())" | Out-File $LogFile -Append
}

# RQ5
if (-not $SkipRQ5) {
    Write-Host ""
    Write-Host "========== RQ5: 超参数影响 ==========" -ForegroundColor Yellow
    $RQ5Start = Get-Date
    
    & "$PSScriptRoot\run_rq5.ps1" -OutputDir $OutputDir -DataDir $DataDir -ModelsDir $ModelsDir -Seed $Seed
    
    $RQ5End = Get-Date
    $RQ5Duration = $RQ5End - $RQ5Start
    $ExperimentResults["RQ5"] = @{
        "Status" = if ($LASTEXITCODE -eq 0) { "Success" } else { "Failed" }
        "Duration" = $RQ5Duration.ToString()
    }
    "RQ5: $($ExperimentResults['RQ5'].Status), 耗时: $($RQ5Duration.ToString())" | Out-File $LogFile -Append
}

# RQ6
if (-not $SkipRQ6) {
    Write-Host ""
    Write-Host "========== RQ6: 超越语法能力 ==========" -ForegroundColor Yellow
    $RQ6Start = Get-Date
    
    & "$PSScriptRoot\run_rq6.ps1" -OutputDir $OutputDir -DataDir $DataDir -ModelsDir $ModelsDir -Seed $Seed
    
    $RQ6End = Get-Date
    $RQ6Duration = $RQ6End - $RQ6Start
    $ExperimentResults["RQ6"] = @{
        "Status" = if ($LASTEXITCODE -eq 0) { "Success" } else { "Failed" }
        "Duration" = $RQ6Duration.ToString()
    }
    "RQ6: $($ExperimentResults['RQ6'].Status), 耗时: $($RQ6Duration.ToString())" | Out-File $LogFile -Append
}

# 生成图表
if ($GenerateFigures) {
    Write-Host ""
    Write-Host "========== 生成可视化图表 ==========" -ForegroundColor Yellow
    
    $FiguresDir = Join-Path $OutputDir "figures"
    
    $Command = "python experiments/analysis.py --results_dir `"$OutputDir`" --output_dir `"$FiguresDir`" --rq all"
    Invoke-Expression "conda activate waf-llm; $Command"
    
    # 生成报告
    $Command = "python experiments/analysis.py --results_dir `"$OutputDir`" --output_dir `"$FiguresDir`" --report"
    Invoke-Expression "conda activate waf-llm; $Command"
}

# 汇总
$EndTime = Get-Date
$TotalDuration = $EndTime - $StartTime

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "              实验完成!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "总耗时: $($TotalDuration.ToString())"
Write-Host ""
Write-Host "实验结果:" -ForegroundColor Yellow
foreach ($rq in $ExperimentResults.Keys | Sort-Object) {
    $status = $ExperimentResults[$rq].Status
    $duration = $ExperimentResults[$rq].Duration
    $color = if ($status -eq "Success") { "Green" } else { "Red" }
    Write-Host "  $rq : $status (耗时: $duration)" -ForegroundColor $color
}
Write-Host ""
Write-Host "结果保存在: $OutputDir" -ForegroundColor Cyan
Write-Host "日志文件: $LogFile" -ForegroundColor Cyan

# 写入最终日志
"" | Out-File $LogFile -Append
"结束时间: $EndTime" | Out-File $LogFile -Append
"总耗时: $($TotalDuration.ToString())" | Out-File $LogFile -Append
