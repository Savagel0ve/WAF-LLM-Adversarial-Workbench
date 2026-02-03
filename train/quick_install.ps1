# 快速安装依赖修复脚本

Write-Host "修复依赖版本冲突..." -ForegroundColor Yellow

# 降级huggingface-hub
pip install "huggingface-hub>=0.30.0,<1.0" --force-reinstall

# 确保transformers版本正确
pip install "transformers>=4.30.0,<4.50.0" --force-reinstall

Write-Host "`n验证安装..." -ForegroundColor Yellow
python -c "from transformers import GPT2Tokenizer; print('✅ Transformers工作正常')"

Write-Host "`n完成!" -ForegroundColor Green
