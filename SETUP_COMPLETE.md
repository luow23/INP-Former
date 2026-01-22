# 🎯 INP-Former Print Quality Detection - HOÀN THÀNH

## ✅ Tất Cả Đã Sẵn Sàng

Dự án của bạn đã được setup hoàn chỉnh với tất cả các công cụ cần thiết!

---

## 📦 Tóm Tắt Setup

| Thành Phần | Trạng Thái | Chi Tiết |
|-----------|-----------|---------|
| **Repository** | ✅ | INP-Former từ luow23/INP-Former |
| **Python Env** | ✅ | Virtual Environment 3.12.3 |
| **PyTorch** | ✅ | 2.10.0+cu128 (GPU Ready) |
| **GPU** | ✅ | RTX 3060 (12GB VRAM) |
| **Dependencies** | ✅ | TimM, Kornia, scikit-learn, v.v. |
| **Dữ Liệu** | ✅ | 48 ảnh mẫu (40 train, 8 test) |
| **Scripts** | ✅ | Training, Inference, Data Prep |

---

## 🚀 3 Bước Nhanh

### BƯỚC 1: Training (2-5 phút)
```bash
cd /home/gabap/Project/INP-Former-OCV
.venv/bin/python train_print_quality.py --epochs 30 --batch_size 8
```

**Output**: Model checkpoints trong `print_quality_output/checkpoints/`

### BƯỚC 2: Inference (1 phút)
```bash
CHECKPOINT=$(ls -t print_quality_output/checkpoints/model_*.pth | head -1)
.venv/bin/python inference_print_quality.py \
    --model_path "$CHECKPOINT" \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode --output_file results.json
```

**Output**: `results.json` với chất lượng từng mẫu in

### BƯỚC 3: Xem Kết Quả
```bash
cat results.json
```

---

## 📋 Files Đã Tạo

### 🔴 Scripts Python

| File | Kích Thước | Mục Đích |
|------|-----------|---------|
| `train_print_quality.py` | 12KB | Training mô hình chính |
| `inference_print_quality.py` | 9.7KB | Kiểm tra chất lượng mẫu in |
| `prepare_print_quality_data.py` | 7.1KB | Chuẩn bị & tổ chức dữ liệu |

### 📘 Hướng Dẫn

| File | Nội Dung |
|------|---------|
| `README_PRINT_QUALITY.md` | Quick start & commands |
| `PRINT_QUALITY_GUIDE.md` | Hướng dẫn chi tiết đầy đủ |
| `SETUP_SUMMARY_VN.md` | Tóm tắt setup (Tiếng Việt) |

### 🔧 Tiện Ích

| File | Mục Đích |
|------|---------|
| `QUICKREF.sh` | Quick reference tất cả lệnh |
| `run_training.sh` | Tự động chạy training |
| `check_setup.sh` | Kiểm tra setup hiện tại |

---

## 📂 Cấu Trúc Dữ Liệu

```
/home/gabap/Project/INP-Former-OCV/
│
├── 📦 TRAINING DATA
│   └── print_quality_data/
│       ├── train/good_print/      (40 ảnh)
│       └── test/good_print/       (8 ảnh)
│
├── 📊 OUTPUT (Sau Training)
│   └── print_quality_output/
│       ├── checkpoints/           (*.pth models)
│       └── logs/                  (training logs)
│
├── 🐍 PYTHON SCRIPTS
│   ├── train_print_quality.py
│   ├── inference_print_quality.py
│   └── prepare_print_quality_data.py
│
├── 📖 DOCUMENTATION
│   ├── README_PRINT_QUALITY.md
│   ├── PRINT_QUALITY_GUIDE.md
│   └── SETUP_SUMMARY_VN.md
│
└── 🔧 UTILITIES
    ├── QUICKREF.sh
    ├── run_training.sh
    └── check_setup.sh
```

---

## 💡 Các Use Cases

### 1️⃣ Kiểm Tra Chất Lượng In Cơ Bản
```bash
# Single image
.venv/bin/python inference_print_quality.py \
    --model_path model.pth \
    --image_path sample.jpg
```

### 2️⃣ Kiểm Tra Lô Mẫu In
```bash
# Directory
.venv/bin/python inference_print_quality.py \
    --model_path model.pth \
    --image_path ./print_samples/ \
    --batch_mode --output_file report.json
```

### 3️⃣ Sử Dụng Dữ Liệu Thực
```bash
# Chuẩn bị dữ liệu từ ảnh thực
.venv/bin/python prepare_print_quality_data.py \
    --source_dir /path/to/real/prints

# Train lại
.venv/bin/python train_print_quality.py --epochs 100
```

### 4️⃣ Tối Ưu Hóa Hiệu Suất
```bash
# Với GPU hạn chế
.venv/bin/python train_print_quality.py \
    --batch_size 4 \
    --encoder dino_vits14

# Với dữ liệu lớn
.venv/bin/python train_print_quality.py \
    --epochs 200 \
    --learning_rate 0.00005
```

---

## 📊 Hiểu Kết Quả

### Quality Score Giải Thích

```
Score < 0.5    → Mẫu in bình thường ✓ (GOOD)
Score = 0.5    → Biên giới (BORDERLINE)
Score > 0.5    → Mẫu in có vấn đề ✗ (DEFECTIVE)
```

### Ví Dụ Output

```json
{
  "predictions": [
    {"image": "sample_1.png", "quality_score": 0.0856, "status": "good"},
    {"image": "sample_2.png", "quality_score": 0.1234, "status": "good"},
    {"image": "sample_3.png", "quality_score": 0.7845, "status": "defective"}
  ],
  "statistics": {
    "mean_score": 0.3245,
    "good_count": 18,
    "defective_count": 2
  }
}
```

---

## 🎛️ Tùy Chỉnh Nâng Cao

### Thay Đổi Model Size

```bash
# Nhẹ & Nhanh (recommended cho GPU nhỏ)
.venv/bin/python train_print_quality.py --encoder dino_vits14

# Cân Bằng
.venv/bin/python train_print_quality.py --encoder dino_base14

# Nặng & Tốt (cần GPU 24GB+)
.venv/bin/python train_print_quality.py --encoder dino_large14
```

### Thay Đổi Tham Số Training

```bash
.venv/bin/python train_print_quality.py \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.00005 \
    --warmup_epochs 20 \
    --inp_num 30
```

### Tùy Chỉnh Ngưỡng Quality

Chỉnh sửa threshold trong `inference_print_quality.py`:
```python
results['predictions'].append({
    'image': str(image_path),
    'quality_score': quality_score,
    'status': 'good' if quality_score < 0.5 else 'defective'  # ← Thay đổi 0.5
})
```

---

## 🔍 Troubleshooting

| Vấn Đề | Giải Pháp |
|--------|----------|
| GPU out of memory | Giảm `--batch_size` xuống 2-4 |
| Model không hội tụ | Tăng `--epochs` hoặc giảm `--learning_rate` |
| Chậm training | Dùng `--encoder dino_vits14` (nhỏ hơn) |
| No data found | Chạy `prepare_print_quality_data.py` |

---

## 📚 Tài Liệu Tham Khảo

### Papers
- [INP-Former CVPR 2025](https://arxiv.org/pdf/2503.02424)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [DINO: Self-Supervised ViT](https://arxiv.org/abs/2104.14294)

### Datasets
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)
- [Real-IAD](https://github.com/gaurav-65)

---

## ✨ Kế Tiếp

### Ngay Bây Giờ
1. ✅ Setup hoàn thành
2. ⏭️ **Training model** (3-5 phút)
3. ⏭️ **Kiểm tra chất lượng** (1 phút)
4. ⏭️ **Xem kết quả** (1 phút)

### Trong Tương Lai
- Thêm dữ liệu mẫu in thực
- Fine-tune model với dữ liệu chuyên biệt
- Deploy model vào production
- Tích hợp vào quy trình QA tự động

---

## 🚀 Start Now!

```bash
# 1. Training
cd /home/gabap/Project/INP-Former-OCV
.venv/bin/python train_print_quality.py --epochs 30

# 2. Inference
CHECKPOINT=$(ls -t print_quality_output/checkpoints/model_*.pth | head -1)
.venv/bin/python inference_print_quality.py \
    --model_path "$CHECKPOINT" \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode --output_file results.json

# 3. View Results
cat results.json
```

---

## 📞 Hỗ Trợ

Xem chi tiết trong:
- `README_PRINT_QUALITY.md` - Quick start
- `PRINT_QUALITY_GUIDE.md` - Hướng dẫn đầy đủ
- `SETUP_SUMMARY_VN.md` - Tóm tắt setup (Tiếng Việt)

---

**🎉 Chúc bạn training thành công!**

*Setup hoàn thành lúc: 22/01/2026 15:22*
