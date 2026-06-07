# 🌿 AI Plant Doctor - Plant Disease Detection

Hệ thống Thị giác Máy tính (Computer Vision) hỗ trợ chẩn đoán 38 loại bệnh trên cây trồng thông qua hình ảnh. Dự án ứng dụng kiến trúc **Vision Transformer (ViT)** kết hợp với công cụ giải thích mô hình **EigenCAM** và trợ lý ảo AI để đưa ra phác đồ điều trị chi tiết.

## ✨ Tính năng nổi bật
* **Nhận diện chính xác 38 lớp bệnh:** Hỗ trợ đa dạng các loại cây trồng như Táo, Cà chua, Ngô, Nho, Đào, Khoai tây... 
* **Bản đồ nhiệt (Heatmap) với EigenCAM:** Không chỉ đưa ra kết quả, hệ thống còn "chỉ điểm" chính xác vị trí vết bệnh trên lá, giúp giải thích quyết định của AI một cách minh bạch.
* **Trợ lý Nông nghiệp AI (LLM):** Tự động phân tích kết quả và xuất ra phác đồ điều trị, cách phòng ngừa bệnh bằng ngôn ngữ tự nhiên.
* **Giao diện thân thiện:** Được xây dựng bằng Gradio, cho phép người dùng kéo thả ảnh và nhận kết quả trực quan ngay trên trình duyệt.

---

## 🗂️ Tập dữ liệu (Dataset)
Hệ thống sử dụng dữ liệu được tổng hợp từ 2 nguồn chính để đảm bảo khả năng nhận diện tốt trong cả điều kiện tiêu chuẩn lẫn môi trường thực tế phức tạp:
* **PlantVillage:** Tập dữ liệu gồm các ảnh lá cây được chụp trong điều kiện phòng thí nghiệm (studio), nền đơn sắc. Đóng vai trò là dữ liệu nền tảng giúp mô hình học các nếp nhăn và vết bệnh rõ ràng.
* **PlantDoc:** Tập dữ liệu gồm các ảnh chụp ngoài môi trường thực tế (in the wild), chứa nhiều nhiễu như ánh sáng phức tạp, lá cây chồng chéo, bóng râm. Việc thêm PlantDoc giúp mô hình tăng mạnh tính khái quát hóa (generalization) khi triển khai thực tế.

**Chiến lược lấy mẫu (Sampling Strategy):**
Do sự chênh lệch lớn về số lượng ảnh giữa các lớp và 2 nguồn dữ liệu, hệ thống sử dụng **WeightedRandomSampler**:
* Cân bằng động giữa 38 lớp bệnh (Class-balanced).
* **Domain Prioritize:** Ưu tiên bốc thăm dữ liệu từ PlantDoc với trọng số **x10** để ép mô hình tập trung học các đặc trưng khó từ môi trường thực tế.

---

## ⚙️ Quy trình Huấn luyện (Training Pipeline)
Quy trình huấn luyện được thiết kế tối ưu cho kiến trúc Transformer với các chiến lược Augmentation phân tầng:

* **1. Data Augmentation theo Domain:**
  * **PlantVillage:** Áp dụng Augmentation nhẹ nhàng (Random Resized Crop, Flip, Rotation 20°, Color Jitter) để giữ nguyên cấu trúc ảnh gốc.
  * **PlantDoc:** Áp dụng Augmentation mạnh tay hơn (Perspective Distortion, Gaussian Blur, Rotation 30°) nhằm bù đắp sự thiếu hụt số lượng dữ liệu và chống overfitting.
* **2. Tiền xử lý (Preprocessing):** Mọi hình ảnh đều được thay đổi kích thước về `224x224` và đi qua `ViTImageProcessor` (`google/vit-base-patch16-224`) để chuẩn hóa vector đầu vào.
* **3. Kiến trúc Mô hình:** Fine-tuning mô hình **Vision Transformer (ViT)**. Khác với CNN truyền thống, ViT chia hình ảnh thành các "patches" (mảnh nhỏ) và sử dụng cơ chế Self-Attention để tìm ra mối tương quan toàn cục giữa vết bệnh và cấu trúc lá.

---

## 📂 Cấu trúc dự án
```text
📦 Plant-Disease-Detection
 ┣ 📂 notebook/          # Chứa các file Jupyter Notebook (EDA, phân tích dữ liệu)
 ┣ 📂 output/            # Chứa các biểu đồ EDA, hình ảnh test và kết quả đánh giá mô hình
 ┣ 📂 src/               # Mã nguồn chính của hệ thống
 ┃ ┣ 📜 app.py               # File giao diện chính (Gradio Web UI)
 ┃ ┣ 📜 dataset.py           # Tiền xử lý dữ liệu, Augmentation và DataLoader
 ┃ ┣ 📜 gradcam.py           # Kỹ thuật EigenCAM tạo bản đồ nhiệt
 ┃ ┣ 📜 inference.py         # Hàm dự đoán chính của mô hình ViT
 ┃ ┣ 📜 LLM_solution.py      # Module tích hợp LLM xuất phác đồ điều trị
 ┃ ┣ 📜 model.py             # Khởi tạo kiến trúc Vision Transformer
 ┃ ┗ 📜 train.py             # Script huấn luyện mô hình
 ┣ 📜 .gitignore         # File cấu hình bỏ qua các file nặng/tạm của Git
 ┣ 📜 requirements.txt   # Danh sách thư viện cần thiết
 ┗ 📜 README.md          # Tài liệu giới thiệu dự án
