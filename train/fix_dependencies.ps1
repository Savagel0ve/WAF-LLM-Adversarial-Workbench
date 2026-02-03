# Fix dependency version conflicts

Write-Host "Fixing dependency conflicts..." -ForegroundColor Yellow

# Downgrade huggingface-hub
Write-Host "Installing huggingface-hub 0.24.0..." -ForegroundColor Cyan
pip install huggingface-hub==0.24.0 --force-reinstall

# Reinstall transformers
Write-Host "Installing transformers 4.40.0..." -ForegroundColor Cyan
pip install transformers==4.40.0 --force-reinstall

Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python -c "from transformers import GPT2Tokenizer; print('OK: Transformers works!')"

Write-Host "`nDone!" -ForegroundColor Green
