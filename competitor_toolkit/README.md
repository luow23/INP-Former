# 选手工具包

本工具包提供数据加载、本地验证、提交打包三个功能模块，帮助选手完成从训练到提交的完整流程。

本比赛分为 **A、B 两轮**，工具包代码自动适配，无需修改：

| | A 轮 | B 轮 |
|---|---|---|
| 类别数 | 50（全部为 seen） | 100（50 seen + 50 zero-shot） |
| 标准答案 | `standard_A.zip` | `standard_B.zip` |
| 提交文件 | `my_submission.zip` | `my_submission_B.zip` |
| 评分公式 | `0.3*S_cls + 0.7*S_seg` | `0.3*S_cls + 0.5*S_seg + 0.2*S_zs` |

所有脚本会自动检测是否存在 zero-shot 类别并切换评分公式，**两轮使用方式完全相同**。

## 目录结构

```
competitor_toolkit/
  dataset.py           # 数据集加载 & 提交构建
  metrics_local.py     # 本地指标计算（与天池评分口径一致）
  make_submission.py   # 打包提交 zip
  requirements.txt     # Python 依赖
```

## 安装依赖

```bash
pip install numpy==1.24.3 pandas==2.0.3 Pillow==9.5.0 scikit-learn==1.3.0
```

## 数据格式

### 训练/验证数据（含标签）

```
standard/
  ground_truth.csv             # group_folder,label
  masks/
    <category>/<sample_id>/
      0_mask.png               # 视角0
      1_mask.png               # 视角1
      2_mask.png               # 视角2
      3_mask.png               # 视角3
      4_mask.png               # 视角4
```

- `ground_truth.csv` 格式：`group_folder,label`，label 为 0（正常）或 1（异常）
- mask 为 448x448 灰度图，像素值越大表示该位置越异常

### 提交文件格式

```
submission.zip
  submission.csv               # group_folder,anomaly_score
  predicted_masks/
    <category>/<sample_id>/
      0_mask.png ~ 4_mask.png  # 448x448 灰度图
```

- `anomaly_score`：样本级异常分数，越大越异常
- 每个样本必须提交 5 个视角的 mask（448x448 灰度图）

## 使用方法

### 1. 加载数据集

```python
from dataset import AnomalyDataset

# 从目录加载（解压后的标准答案）
ds = AnomalyDataset.from_dir("./standard")

# 或从 zip 加载
ds = AnomalyDataset.from_zip("standard_A.zip")

# 基本信息
print("类别数:", len(ds.categories))        # 50
print("样本数:", len(ds.all_samples))       # 750
print("类别:", ds.categories)

# 加载一个样本的5视角图片
images = ds.load_images("3_adapter/S0001")  # shape=(5, 448, 448)
label = ds.get_label("3_adapter/S0001")     # 0 或 1

# 获取某个类别所有样本
samples = ds.get_category_samples("3_adapter")
```

### 2. 构建提交结果

```python
from dataset import SubmissionBuilder
import numpy as np

builder = SubmissionBuilder()

# 为每个样本添加预测结果
for group_folder in test_samples:
    # your_model.predict() 得到 anomaly_score 和 5 个 mask
    anomaly_score = 0.12
    masks = [np.zeros((448, 448), dtype=np.float32) for _ in range(5)]
    builder.add_sample(group_folder, anomaly_score, masks)

# 保存并打包
builder.save("outputs/submission", zip_path="my_submission.zip")
```

### 3. 本地验证

选手可以自行划分验证集来评估模型：

```bash
# 从目录计算
python metrics_local.py \
    --standard-dir path/to/val_ground_truth \
    --submission-dir path/to/val_predictions \
    --out metrics.json

# 从 zip 计算
python metrics_local.py \
    --standard-zip val_standard.zip \
    --submission-zip val_submission.zip \
    --out metrics.json
```

输出示例：

```json
{
  "score": 62.19,
  "I-AUROC": 0.8937,
  "I-AP": 0.9454,
  "P-AUROC": 0.9181,
  "P-AP": 0.2415,
  "P-F1max": 0.3234,
  "S_cls": 0.9195,
  "S_seg": 0.4943
}
```

### 4. 打包提交

如果你已有模型输出的 scores.csv 和 masks 目录：

```bash
python make_submission.py \
    --scores-csv outputs/scores.csv \
    --mask-root outputs/masks \
    --out-dir outputs/submission \
    --zip my_submission.zip
```

`scores.csv` 格式：

```csv
group_folder,anomaly_score
3_adapter/S0001,0.1169
3_adapter/S0002,0.1210
```

`mask-root` 目录结构：

```
outputs/masks/
  3_adapter/
    S0001/
      0_mask.png
      1_mask.png
      2_mask.png
      3_mask.png
      4_mask.png
```

## 评分规则

最终分数计算公式：

- **无 zero-shot 类别时**：`score = (0.3 * S_cls + 0.7 * S_seg) * 100`
- **有 zero-shot 类别时**：`score = (0.3 * S_cls + 0.5 * S_seg + 0.2 * S_zs) * 100`

其中：

- `S_cls = mean(I-AUROC, I-AP)` — 图像级分类指标均值（seen 类别）
- `S_seg = mean(P-AUROC, P-AP, P-F1max)` — 像素级分割指标均值（所有类别）
- `S_zs` = zero-shot 类别的 5 项指标均值

### 指标说明

| 指标 | 含义 |
|---|---|
| I-AUROC | 图像级 ROC-AUC，判断样本是否异常 |
| I-AP | 图像级 Average Precision |
| P-AUROC | 像素级 ROC-AUC，评估异常定位能力 |
| P-AP | 像素级 Average Precision |
| P-F1max | 像素级 precision-recall 曲线上最大 F1 值 |
