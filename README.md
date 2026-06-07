# 🌿 AI Plant Doctor — Plant Disease Detection

Hệ thống **Computer Vision** chẩn đoán bệnh trên **14 loài cây trồng** từ ảnh lá cây.  
Sử dụng **Vision Transformer (ViT)** + **EigenCAM** để giải thích vùng bệnh + **Gemini AI** để đề xuất giải pháp điều trị.

🔗 **Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/ManhPhat2104/Leaf-Disease-Detection) | 📄 **Báo cáo:** TDTU — Computer Vision

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 🔍 Phân loại bệnh | ViT-B/16 fine-tuned, 38 classes, Top-3 confidence |
| 🔥 EigenCAM Heatmap | Highlight vùng bệnh trên lá, giải thích quyết định AI |
| 🤖 Trợ lý AI | Gemini đưa ra nguyên nhân, cách xử lý, phòng ngừa |
| 📊 Confidence chart | Biểu đồ phân phối xác suất top predictions |
| ⚡ Streaming UI | Kết quả hiển thị từng bước, không chờ toàn bộ |

---

## 🗂️ Dataset

| Dataset | Số ảnh | Loại | Vai trò |
|---|---|---|---|
| **PlantVillage** | ~54,305 | Studio (nền đơn sắc) | Nền tảng học đặc trưng bệnh |
| **PlantDoc** | ~2,569 | Thực tế (ngoài đồng) | Tăng khả năng tổng quát hóa |

**Chiến lược gộp data:**
- PlantDoc split (train/valid/test) gộp vào đúng tập tương ứng — **không có data leakage**
- `WeightedRandomSampler`: cân bằng class + **ưu tiên PlantDoc x10** để bù chênh lệch domain
- Augmentation **phân tầng theo domain**: PlantDoc dùng augmentation mạnh hơn PlantVillage

---

## 🧠 Kiến trúc mô hình

```
Input ảnh lá (224×224)
        ↓
ViTImageProcessor (normalize ImageNet)
        ↓
ViT-B/16 (google/vit-base-patch16-224)
   - 12 Transformer blocks
   - 16×16 patch size · 768 hidden dim · 12 attention heads
        ↓
Classification Head (Linear 768 → 38)
        ↓
Top-3 Predictions + EigenCAM Heatmap + Gemini Solution
```

**Fine-tuning 3 phases:**

| Phase | Epochs | Layers train | Learning rate |
|---|---|---|---|
| 1 | 3 | Chỉ classifier head | 1e-3 |
| 2 | 7 | 4 blocks cuối + head | 1e-4 (layer-wise decay) |
| 3 | 15 | Toàn bộ model | 1e-5 / 1e-4 |

---

## 📂 Cấu trúc dự án

```
📦 plant-disease-detection/
 ┣ 📂 src/
 ┃ ┣ 📜 model.py              # ViT build_model() + load_checkpoint() + CLASS_NAMES
 ┃ ┣ 📜 dataset.py            # LeafDiseaseDataset, read_plantdoc(), get_dataloader()
 ┃ ┣ 📜 train.py              # 3-phase training, early stopping, biểu đồ
 ┃ ┣ 📜 inference.py          # Singleton get_model(), predict(), load_image()
 ┃ ┣ 📜 gradcam.py            # EigenCAM, ViTGradCAMWrapper, analyze_image()
 ┃ ┣ 📜 LLM_solution.py       # Gemini API, get_treatment_solution()
 ┃ ┗ 📜 app.py                # Gradio UI, streaming output
 ┣ 📂 notebook/
 ┃ ┗ 📓 eda.ipynb             # EDA: phân phối class, domain gap, augmentation viz
 ┣ 📂 output/
 ┃ ┣ 🖼️ loss_curve.png
 ┃ ┣ 🖼️ metrics_curve.png
 ┃ ┗ 🖼️ confusion_matrix.png
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```
## 🚀 Cài đặt & Chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Key (Gemini)

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux / Mac
export GEMINI_API_KEY=your_api_key_here
```

Lấy API key miễn phí tại: [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 3. Train mô hình

```bash
# Sửa đường dẫn dataset trong CONFIG trước
python src/train.py
```

### 4. Chạy app

```bash
python src/app.py
# → http://localhost:7860
```
---

## 📊 Kết quả thực nghiệm
| Experiment | Dataset | Accuracy | F1-Macro |
|---|---|---|---|
| + PlantDoc | PV + PlantDoc | 98.8%| 98.4% |
---

## 🛠️ Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Model | `transformers` — ViTForImageClassification |
| Explainability | `pytorch-grad-cam` — EigenCAM |
| Training | PyTorch + AMP (mixed precision) |
| Preprocessing | `ViTImageProcessor` (HuggingFace) |
| LLM | Google Gemini 2.5 Flash Lite |
| UI | Gradio 4.x |
| Deploy | Hugging Face Spaces |

---

## ⚠️ Giới hạn
- Dataset PlantVillage là ảnh studio — độ chính xác giảm với ảnh thực tế nhiều lá chồng chéo
- Nên chụp **1 lá rõ ràng** chiếm 70–80% khung hình để đạt kết quả tốt nhất
---
