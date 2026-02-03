# Quick WAF Connection Test
# Verify WAF is working properly

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "WAF Connection Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$WAF_URL = "http://localhost:8081"

Write-Host "`nWAF URL: $WAF_URL" -ForegroundColor White

# Test SQLi
Write-Host "`nTesting SQLi..." -ForegroundColor Yellow
python train\test_waf_connection.py --waf_url $WAF_URL --attack_type sqli

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSQLi Test PASSED" -ForegroundColor Green
} else {
    Write-Host "`nSQLi Test FAILED" -ForegroundColor Red
    exit 1
}

# Test XSS
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing XSS..." -ForegroundColor Yellow
python train\test_waf_connection.py --waf_url $WAF_URL --attack_type xss

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nXSS Test PASSED" -ForegroundColor Green
} else {
    Write-Host "`nXSS Test FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Tests PASSED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nWAF is working. Ready to train reward model." -ForegroundColor White
