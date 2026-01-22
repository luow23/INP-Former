# 🖨️ INP-Former Print Quality Detection

Training INP-Former để kiểm tra **chất lượng mẫu in** mà không cần quan tâm nội dung bên trong.

## ⚡ Quick Start (5 phút)

### 1️⃣ Training

```bash
cd /home/gabap/Project/INP-Former-OCV
.venv/bin/python train_print_quality.py --epochs 30 --batch_size 8
```

### 2️⃣ Inference

```bash
# Tìm checkpoint mới nhất
CHECKPOINT=$(ls -t print_quality_output/checkpoints/model_*.pth | head -1)

# Kiểm tra chất lượng mẫu in
.venv/bin/python inference_print_quality.py \
    --model_path "$CHECKPOINT" \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode --output_file results.json
```

### 3️⃣ Xem Kết Quả

```bash
cat results.json
```

---

## 📚 Hướng Dẫn Chi Tiết

Xem [PRINT_QUALITY_GUIDE.md](PRINT_QUALITY_GUIDE.md) cho hướng dẫn đầy đủ.

Xem [SETUP_SUMMARY_VN.md](SETUP_SUMMARY_VN.md) để hiểu setup hiện tại.

---

## 📦 Các Scripts Có Sẵn

| Script | Mục Đích |
|--------|----------|
| `train_print_quality.py` | Training mô hình |
| `inference_print_quality.py` | Kiểm tra chất lượng |
| `prepare_print_quality_data.py` | Chuẩn bị dữ liệu |
| `run_training.sh` | Tự động chạy training |
| `check_setup.sh` | Kiểm tra setup |

---

## 🎯 Cấu Trúc Dữ Liệu

```
print_quality_data/
├── train/
│   └── good_print/           (40 ảnh mẫu tốt)
└── test/
    └── good_print/           (8 ảnh kiểm tra)
```

**Bạn có thể thêm dữ liệu thực**:
```bash
.venv/bin/python prepare_print_quality_data.py \
    --source_dir /path/to/your/print/images
```

---

## 💻 GPU: RTX 3060 (12GB VRAM) ✓

Đã được phát hiện và sẵn sàng sử dụng!

---

## 🚀 Lệnh Hay Dùng

```bash
# Training với tuỳ chỉnh
.venv/bin/python train_print_quality.py \
    --data_path ./print_quality_data \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --encoder dino_vits14

# Inference trên thư mục
.venv/bin/python inference_print_quality.py \
    --model_path ./print_quality_output/checkpoints/model_epoch_050_loss_xxxx.pth \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode

# Xem GPU
nvidia-smi

# Xem logs
tail -f ./print_quality_output/logs/training_*.log
```

---

## 📊 Hiểu Kết Quả

**Quality Score**:
- **Thấp (< 0.5)**: Mẫu in tốt ✓
- **Cao (> 0.5)**: Mẫu in có vấn đề ✗

Mô hình học từ các ảnh bình thường trong training và phát hiện những gì khác thường (khuyết tật, độ rõ ràng kém, thẩm thấu ink tệ).

---

## 📝 Thêm Dữ Liệu Thực

```bash
# Chuẩn bị dữ liệu từ ảnh thực của bạn
.venv/bin/python prepare_print_quality_data.py \
    --source_dir /đường/dẫn/đến/ảnh \
    --output_dir ./print_quality_data \
    --train_ratio 0.8

# Training lại với dữ liệu mới
.venv/bin/python train_print_quality.py --epochs 50
```

---

## 🎓 Tiếp Theo

1. **Thêm dữ liệu mẫu in thực** của bạn vào `print_quality_data/train/good_print/`
2. **Train lại** model với dữ liệu mới
3. **Kiểm tra** chất lượng các mẫu in khác

---

**Chúc bạn training thành công! 🎉**

---

*INP-Former: Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection (CVPR 2025)*
