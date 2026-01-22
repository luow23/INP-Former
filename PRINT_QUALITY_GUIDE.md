# INP-Former for Print Quality Detection

Hướng dẫn đầy đủ để training và sử dụng INP-Former nhằm kiểm tra **chất lượng mẫu in** mà không cần quan tâm đến nội dung bên trong.

## 📋 Tổng Quan

INP-Former là một mô hình sử dụng Vision Transformer để phát hiện anomaly (bất thường) trong ảnh. Trong trường hợp này, chúng ta sử dụng nó để:

- **Học đặc trưng chất lượng in**: Từ những mẫu in bình thường
- **Phát hiện khuyết tật**: Độ rõ ràng, thẩm thấu ink, khuyết tật vật lý
- **Không cần biết nội dung**: Mô hình tập trung vào chất lượng bề mặt, không phần nội dung

## 🚀 Bắt Đầu Nhanh

### 1. Chuẩn Bị Dữ Liệu

#### Option A: Tạo dữ liệu mẫu (cho testing)
```bash
cd /home/gabap/Project/INP-Former-OCV

# Tạo 20 ảnh mẫu tổng hợp
python prepare_print_quality_data.py --create_samples --num_samples 20
```

**Kết quả:**
```
print_quality_data/
├── train/
│   └── good_print/
│       ├── sample_train_0000.png
│       ├── sample_train_0001.png
│       └── ...
└── test/
    └── good_print/
        ├── sample_test_0000.png
        └── ...
```

#### Option B: Sử dụng dữ liệu thực của bạn
```bash
python prepare_print_quality_data.py --source_dir /path/to/your/print/images --train_ratio 0.8
```

**Yêu cầu cấu trúc thư mục:**
```
your_print_images/
├── image1.jpg
├── image2.jpg
├── image3.png
└── ...
```

#### Option C: Tổ chức thủ công

Tạo cấu trúc như sau:
```
print_quality_data/
├── train/
│   ├── good_print/          (mẫu in tốt - training)
│   │   ├── sample_1.jpg
│   │   ├── sample_2.jpg
│   │   └── ...
│   └── defective_print/     (optional - mẫu lỗi, để tham khảo)
│       ├── defect_1.jpg
│       └── ...
└── test/
    ├── good_print/
    │   ├── test_1.jpg
    │   └── ...
    └── defective_print/     (optional)
        └── ...
```

### 2. Cài Đặt Dependencies

```bash
cd /home/gabap/Project/INP-Former-OCV

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Nếu có lỗi, cài từng package:
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install timm kornia scikit-image scikit-learn
```

**Lưu ý**: Python 3.8-3.11 được recommend. Bạn đang dùng Python 3.12, có thể gặp vài cảnh báo nhưng vẫn hoạt động.

### 3. Training Mô Hình

```bash
# Training cơ bản (dùng default parameters)
python train_print_quality.py \
    --data_path ./print_quality_data \
    --output_dir ./print_quality_output \
    --epochs 100 \
    --batch_size 8 \
    --encoder dino_vits14

# Hoặc với GPU hạn chế (batch size nhỏ hơn)
python train_print_quality.py \
    --data_path ./print_quality_data \
    --output_dir ./print_quality_output \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --encoder dino_vits14
```

**Các tham số quan trọng:**

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `--epochs` | 100 | Số epoch training |
| `--batch_size` | 8 | Kích thước batch |
| `--learning_rate` | 0.0001 | Learning rate |
| `--encoder` | dino_vits14 | ViT encoder (`dino_vits14`, `dino_base14`, `dino_large14`) |
| `--input_size` | 224 | Kích thước ảnh input |
| `--warmup_epochs` | 10 | Số epoch warmup |

**Lựa chọn Encoder:**
- `dino_vits14`: Nhẹ, nhanh (recommended cho GPU nhỏ)
- `dino_base14`: Cân bằng hiệu suất/tốc độ
- `dino_large14`: Nặng, cần GPU 24GB+ (như RTX 4090)

**Output:**
```
print_quality_output/
├── checkpoints/
│   ├── model_epoch_001_loss_0.1234.pth
│   ├── model_epoch_002_loss_0.1156.pth
│   └── ...
└── logs/
    └── training_20260122_145000.log
```

### 4. Kiểm Tra Chất Lượng Mẫu In

#### Inference trên một ảnh:
```bash
python inference_print_quality.py \
    --model_path ./print_quality_output/checkpoints/model_epoch_050_loss_0.0856.pth \
    --image_path ./test_sample.jpg \
    --encoder dino_vits14
```

#### Inference trên toàn bộ thư mục:
```bash
python inference_print_quality.py \
    --model_path ./print_quality_output/checkpoints/model_epoch_050_loss_0.0856.pth \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode \
    --output_file ./quality_results.json \
    --encoder dino_vits14
```

**Output mẫu:**
```
============================================================
PRINT QUALITY DETECTION REPORT
============================================================

Statistics:
  Mean Quality Score: 0.1245
  Std Dev: 0.0324
  Range: [0.0856, 0.2134]
  Good Samples: 18
  Defective Samples: 2

Predictions (20 total):
------------------------------------------------------------
  sample_test_0000.png       | Score: 0.0856 | good
  sample_test_0001.png       | Score: 0.0912 | good
  sample_test_0002.png       | Score: 0.1856 | defective
  ...

============================================================
```

## 📊 Hiểu Kết Quả

### Quality Score (Điểm Chất Lượng)

- **Thấp (< 0.5)**: Mẫu in tốt, chất lượng bình thường
- **Cao (> 0.5)**: Mẫu in có vấn đề, cần kiểm tra

Score được tính dựa trên **reconstruction error** - mức độ khác biệt giữa:
- Ảnh gốc
- Ảnh được khôi phục lại từ "normal tokens" của mô hình

**Ý tưởng chính**: 
- Mẫu in tốt → Mô hình có thể khôi phục lại tốt → Score thấp
- Mẫu in có lỗi → Mô hình khôi phục không tốt → Score cao

## 🔧 Tùy Chỉnh Nâng Cao

### Thay đổi chiến lược training

**Sửa file `train_print_quality.py`:**

```python
# Thay đổi số lượng INP (Intrinsic Normal Prototypes)
parser.add_argument('--inp_num', type=int, default=20)  # Tăng lên để học thêm pattern

# Thay đổi weight decay
parser.add_argument('--weight_decay', type=float, default=1e-4)
```

### Sử dụng Custom Loss Function

Nếu muốn ưu tiên một loại lỗi cụ thể (e.g., ink saturation vs clarity):

```python
# Thêm vào train_print_quality.py
class PrintQualityLoss(nn.Module):
    def __init__(self, weight_saturation=1.0, weight_clarity=1.0):
        super().__init__()
        self.weight_saturation = weight_saturation
        self.weight_clarity = weight_clarity
    
    def forward(self, reconstruction_error, target):
        # Custom loss logic
        return loss
```

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"
```bash
# Giảm batch size
python train_print_quality.py --batch_size 2 --encoder dino_vits14

# Hoặc sử dụng encoder nhỏ hơn
python train_print_quality.py --encoder dino_vits14
```

### Lỗi: "No data found"
```bash
# Kiểm tra cấu trúc thư mục dữ liệu
ls -la print_quality_data/train/good_print/
ls -la print_quality_data/test/good_print/

# Nếu trống, tạo dữ liệu mẫu
python prepare_print_quality_data.py --create_samples --num_samples 30
```

### Lỗi: "Model checkpoint not found"
```bash
# Kiểm tra path checkpoint
ls -la print_quality_output/checkpoints/

# Sử dụng đường dẫn đầy đủ
python inference_print_quality.py \
    --model_path /home/gabap/Project/INP-Former-OCV/print_quality_output/checkpoints/model_epoch_050_loss_0.0856.pth \
    --image_path ./test_sample.jpg
```

## 📚 Tham Khảo Thêm

### Papers & Resources
- [INP-Former Paper](https://arxiv.org/pdf/2503.02424)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

### Dataset References
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)
- [Real-IAD](https://github.com/gaurav-65)

## 📝 Ví Dụ Quy Trình Hoàn Chỉnh

```bash
# 1. Chuẩn bị dữ liệu
python prepare_print_quality_data.py --create_samples --num_samples 50

# 2. Training
python train_print_quality.py \
    --data_path ./print_quality_data \
    --output_dir ./print_quality_output \
    --epochs 100 \
    --batch_size 8 \
    --encoder dino_vits14

# 3. Inference trên test set
python inference_print_quality.py \
    --model_path ./print_quality_output/checkpoints/model_epoch_100_loss_xxxx.pth \
    --image_path ./print_quality_data/test/good_print/ \
    --batch_mode \
    --output_file ./results.json

# 4. Xem kết quả
cat ./results.json
```

## 🤝 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra logs: `cat print_quality_output/logs/training_*.log`
2. Xem output JSON từ inference
3. Kiểm tra GPU: `nvidia-smi`

---

**Chúc bạn training thành công! 🎉**
