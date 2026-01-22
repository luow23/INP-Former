## 📋 Tóm Tắt Setup INP-Former cho Kiểm Tra Chất Lượng Mẫu In

### ✅ Đã Hoàn Thành

1. **✓ Clone Repository**: INP-Former từ GitHub (luow23/INP-Former)
2. **✓ Cài Đặt Dependencies**: PyTorch 2.0.0, TimM, Kornia, scikit-learn, v.v.
3. **✓ Tạo Scripts Training & Inference**: 
   - `train_print_quality.py` - Script training chính
   - `inference_print_quality.py` - Script kiểm tra chất lượng
   - `prepare_print_quality_data.py` - Chuẩn bị dữ liệu
4. **✓ Chuẩn Bị Dữ Liệu**: Tạo 40 ảnh mẫu tổng hợp
5. **✓ GPU Ready**: Phát hiện RTX 3060 (12GB VRAM)

---

### 🚀 Các Bước Tiếp Theo

#### **BƯỚC 1: Training Mô Hình**

```bash
cd /home/gabap/Project/INP-Former-OCV

# Option A: Chạy script training
bash run_training.sh

# Option B: Chạy lệnh trực tiếp
.venv/bin/python train_print_quality.py \
    --data_path ./print_quality_data \
    --output_dir ./print_quality_output \
    --epochs 30 \
    --batch_size 8 \
    --encoder dino_vits14
```

**Thời gian dự kiến**: ~2-5 phút cho 30 epochs trên RTX 3060

**Output**:
```
print_quality_output/
├── checkpoints/
│   ├── model_epoch_001_loss_0.xxxx.pth
│   ├── model_epoch_002_loss_0.xxxx.pth
│   └── ...
└── logs/
    └── training_YYYYMMDD_HHMMSS.log
```

#### **BƯỚC 2: Kiểm Tra Chất Lượng Mẫu In**

Sau khi training hoàn thành, chạy inference:

```bash
# Trên một ảnh
.venv/bin/python inference_print_quality.py \
    --model_path ./print_quality_output/checkpoints/model_epoch_030_loss_xxxx.pth \
    --image_path ./print_quality_data/test/good_print/sample_test_0000.png

# Trên toàn bộ thư mục test
.venv/bin/python inference_print_quality.py \
    --model_path ./print_quality_output/checkpoints/model_epoch_030_loss_xxxx.pth \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode \
    --output_file ./quality_results.json
```

---

### 📊 Hiểu Kết Quả

**Quality Score**:
- **< 0.5**: Mẫu in bình thường ✓ (good)
- **> 0.5**: Mẫu in có vấn đề ✗ (defective)

**Ví dụ output**:
```
Quality Score: 0.0845
Status: Good (chất lượng in tốt)
```

---

### 📁 Cấu Trúc Thư Mục Hiện Tại

```
/home/gabap/Project/INP-Former-OCV/
├── train_print_quality.py           ← Script training
├── inference_print_quality.py        ← Script inference
├── prepare_print_quality_data.py     ← Chuẩn bị dữ liệu
├── PRINT_QUALITY_GUIDE.md            ← Hướng dẫn chi tiết
├── run_training.sh                   ← Script tự động
├── print_quality_data/               ← Dữ liệu training/test
│   ├── train/
│   │   └── good_print/              (40 ảnh)
│   └── test/
│       └── good_print/              (8 ảnh)
├── print_quality_output/             ← (sẽ tạo sau training)
│   ├── checkpoints/
│   └── logs/
└── [INP-Former repo files...]
```

---

### 🎯 Quy Trình Nhanh (5 phút)

```bash
cd /home/gabap/Project/INP-Former-OCV

# 1. Training (2-5 phút)
.venv/bin/python train_print_quality.py --epochs 30 --batch_size 8

# 2. Inference (1 phút)
# Tìm checkpoint mới nhất
LATEST_CHECKPOINT=$(ls -t print_quality_output/checkpoints/model_*.pth | head -1)

# Chạy inference
.venv/bin/python inference_print_quality.py \
    --model_path "$LATEST_CHECKPOINT" \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode \
    --output_file ./results.json

# 3. Xem kết quả
cat ./results.json
```

---

### 💡 Tùy Chỉnh Cho Dữ Liệu Thực

Nếu bạn có ảnh mẫu in thực:

```bash
# Chuẩn bị dữ liệu
.venv/bin/python prepare_print_quality_data.py \
    --source_dir /path/to/your/print/images \
    --output_dir ./print_quality_data \
    --train_ratio 0.8

# Sau đó training như bình thường
.venv/bin/python train_print_quality.py \
    --data_path ./print_quality_data \
    --epochs 50 \
    --batch_size 8
```

**Cấu trúc dữ liệu cần**:
```
your_print_images/
├── image1.jpg
├── image2.jpg
├── image3.png
└── ... (tối thiểu 20 ảnh cho training)
```

---

### ⚙️ Điều Chỉnh Hiệu Suất

| Nhu Cầu | Cách Làm |
|---------|---------|
| **Training nhanh hơn** | Giảm `--epochs` hoặc `--batch_size` |
| **Kết quả tốt hơn** | Tăng `--epochs` hoặc dữ liệu training |
| **GPU RAM hạn chế** | Giảm `--batch_size` xuống 2-4 |
| **Sử dụng model nhẹ** | Chọn `--encoder dino_vits14` (mặc định) |

---

### 🔍 Debug & Troubleshooting

**Kiểm tra logs**:
```bash
tail -f ./print_quality_output/logs/training_*.log
```

**Kiểm tra dữ liệu**:
```bash
# Số lượng ảnh training
find ./print_quality_data/train -type f | wc -l

# Số lượng ảnh test
find ./print_quality_data/test -type f | wc -l
```

**Kiểm tra GPU**:
```bash
nvidia-smi
```

---

### 📚 Files Quan Trọng Tham Khảo

- `PRINT_QUALITY_GUIDE.md` - Hướng dẫn chi tiết đầy đủ
- `check_setup.sh` - Kiểm tra setup hiện tại
- `models/uad.py` - Code mô hình INP-Former
- `dataset.py` - Code xử lý dữ liệu

---

### ✨ Sẵn Sàng!

Bạn đã sẵn sàng để:
1. **Training** mô hình INP-Former để phát hiện chất lượng in
2. **Inference** trên các mẫu in mới
3. **Đánh giá** chất lượng mà không cần biết nội dung bên trong

**Bước tiếp theo**: Chạy `bash run_training.sh` hoặc các lệnh training ở trên! 🚀
