# 奖励模型训练快速开始

5分钟快速上手奖励模型训练！

## 🚀 快速开始

### 步骤 0: 检查前置条件

```powershell
# 1. 检查预训练模型是否存在
Test-Path .\models\pretrain_sqli_gpt2_small

# 2. 检查 WAF 是否运行
.\test_reward_waf.ps1
```

如果 WAF 测试失败，请启动 WAF：

```powershell
cd waf-bench
docker-compose up -d
```

### 步骤 1: 训练 SQLi 奖励模型（推荐先做这个）

```powershell
.\train_reward_sqli.ps1
```

这个脚本会：
1. ✅ 从 SQLi 数据采样 4000 条
2. ✅ 通过 WAF 测试并打标签（~5-10分钟）
3. ✅ 训练 GPT-2 分类模型（~10-15分钟）
4. ✅ 在测试集上评估

**总耗时**: 约 15-25 分钟

### 步骤 2: 测试模型

```powershell
# 单个 payload 测试
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload "' OR 1=1 --"

# 批量测试
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload_file .\data\processed\sqli\test.txt
```

### 步骤 3: 查看训练结果

```powershell
# 启动 TensorBoard
tensorboard --logdir .\models\reward_sqli\logs

# 在浏览器打开: http://localhost:6006
```

## 📊 预期结果

### 训练指标

| 指标 | SQLi 目标 | XSS 目标 |
|------|-----------|----------|
| AUC | > 99% | > 98% |
| F1-Score | > 95% | > 95% |
| Accuracy | > 95% | > 95% |

### 输出示例

```
测试集结果:
  test_accuracy: 0.9612
  test_f1: 0.9583
  test_precision: 0.9521
  test_recall: 0.9647
  test_auc: 0.9924
```

如果 AUC < 95%，可能需要：
- 增加数据量
- 调整超参数
- 检查数据质量

## 🔧 自定义训练

### 修改采样数量

```powershell
# 快速测试（小数据量）
python train\generate_labeled_data.py `
    --attack_type sqli `
    --input_file .\data\processed\sqli\train.txt `
    --output_dir .\data\labeled `
    --num_samples 1000  # 降低到 1000

# 高质量训练（大数据量）
python train\generate_labeled_data.py `
    --num_samples 8000  # 增加到 8000
```

### 修改训练参数

```powershell
python train\train_reward_model.py `
    --pretrained_model_path .\models\pretrain_sqli_gpt2_small `
    --data_path .\data\labeled `
    --output_dir .\models\reward_sqli_custom `
    --batch_size 16 `       # 降低显存占用
    --epochs 6 `            # 增加训练轮数
    --learning_rate 1e-5    # 降低学习率
```

### 使用不同的 WAF

```powershell
# ModSecurity (默认)
python train\generate_labeled_data.py `
    --waf_url http://localhost:8081

# Naxsi
python train\generate_labeled_data.py `
    --waf_url http://localhost:8082

# 自定义 WAF
python train\generate_labeled_data.py `
    --waf_url http://your-waf-server:port
```

## 📁 输出文件

训练完成后，会生成以下文件：

```
data/labeled/
├── sqli_train.csv          # 训练数据
├── sqli_val.csv            # 验证数据
├── sqli_test.csv           # 测试数据
├── sqli_train_full.json    # 完整信息（含响应时间等）
└── sqli_stats.json         # 数据统计

models/reward_sqli/
├── final_reward_model/     # 最终模型（用于 PPO）
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.json
├── checkpoint-*/           # 训练检查点
├── logs/                   # TensorBoard 日志
└── test_results.json       # 测试结果
```

## 🐛 常见问题

### Q1: WAF 连接失败？

```powershell
# 检查 WAF 状态
cd waf-bench
docker-compose ps

# 重启 WAF
docker-compose restart

# 查看 WAF 日志
docker-compose logs -f
```

### Q2: 显存不足？

```powershell
# 方案1: 降低 batch size
python train\train_reward_model.py --batch_size 16

# 方案2: 不使用 FP16（不推荐）
python train\train_reward_model.py --fp16 false

# 方案3: 减少序列长度
python train\train_reward_model.py --max_length 64
```

### Q3: 训练太慢？

```powershell
# 减少数据量（快速测试）
python train\generate_labeled_data.py --num_samples 1000

# 减少 epoch
python train\train_reward_model.py --epochs 2

# 增加 batch size（如果显存够）
python train\train_reward_model.py --batch_size 64
```

### Q4: 模型性能不佳？

1. **增加数据量**: `--num_samples 8000`
2. **增加训练轮数**: `--epochs 6`
3. **调整学习率**: `--learning_rate 1e-5`
4. **检查数据质量**: 查看 `sqli_stats.json`
5. **平衡数据集**: `--balance_ratio 0.6`

## 🎯 下一步

训练完成后：

### 1. 评估模型

```powershell
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload_file .\data\processed\sqli\test.txt
```

### 2. 在代码中使用

```python
from train.test_reward_model import RewardModelInference

# 加载模型
reward_model = RewardModelInference(
    model_path="./models/reward_sqli/final_reward_model"
)

# 预测
payloads = ["' OR 1=1 --", "UNION SELECT", "normal input"]
probs = reward_model.predict_batch(payloads)

for payload, prob in zip(payloads, probs):
    print(f"{prob:.3f} | {payload}")
```

### 3. 进入 PPO 阶段

奖励模型将用于强化学习：

```python
# 在 PPO 训练中使用
reward_model_path = "./models/reward_sqli/final_reward_model"
```

## 📚 参考文档

- **详细指南**: `README_REWARD_MODEL.md`
- **论文细节**: `GPTFuzzer 论文细节总结.md`
- **训练指南**: `WAF 绕过奖励模型训练指南.md`

## ⚡ 完整流程示例

```powershell
# 1. 测试 WAF
.\test_reward_waf.ps1

# 2. 训练 SQLi 模型
.\train_reward_sqli.ps1

# 3. 训练 XSS 模型
.\train_reward_xss.ps1

# 4. 测试模型
python train\test_reward_model.py `
    --model_path .\models\reward_sqli\final_reward_model `
    --payload "' OR 1=1 --"

# 5. 查看训练日志
tensorboard --logdir .\models\reward_sqli\logs
```

---

**需要帮助？** 

- 查看日志文件: `logs/reward_training.log`
- 提交 Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 参考文档: `README_REWARD_MODEL.md`
