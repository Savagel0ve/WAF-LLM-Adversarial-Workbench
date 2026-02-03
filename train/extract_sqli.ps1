# SQLi Dataset Extraction Script (Windows PowerShell)
# Extract multi-volume zip files

Write-Host "========================================"
Write-Host "SQLi Dataset Extraction Tool"
Write-Host "========================================"

$sqlDir = "gptfuzzer-main\Datasets\SQLi"
$zipFile = "$sqlDir\SQLi_Dataset.zip"
$txtFile = "$sqlDir\SQLi_Dataset.txt"

# Check if already extracted
if (Test-Path $txtFile) {
    $size = (Get-Item $txtFile).Length / 1MB
    Write-Host "`nSQLi_Dataset.txt already exists (Size: $([math]::Round($size, 2)) MB)"
    Write-Host "If you want to re-extract, delete the file first"
    exit 0
}

# Try to find 7-Zip
$7zipPaths = @(
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files (x86)\7-Zip\7z.exe"
)

$7zip = $null
foreach ($path in $7zipPaths) {
    if (Test-Path $path) {
        $7zip = $path
        break
    }
}

if ($7zip) {
    Write-Host "`nFound 7-Zip: $7zip"
    Write-Host "Extracting..."
    
    & $7zip x "$zipFile" -o"$sqlDir" -y
    
    if (Test-Path $txtFile) {
        $size = (Get-Item $txtFile).Length / 1MB
        Write-Host "`nExtraction successful! File size: $([math]::Round($size, 2)) MB"
        Write-Host "`nNext step: python train/prepare_data.py --attack-type sqli"
    } else {
        Write-Host "`nExtraction failed"
    }
} else {
    Write-Host "`nPlease manually extract:"
    Write-Host "1. Install 7-Zip from https://www.7-zip.org/"
    Write-Host "2. Or right-click SQLi_Dataset.zip in $sqlDir and extract"
}
