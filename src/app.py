import os
import torch
import numpy as np
import gradio as gr
from PIL import Image

# Import từ các file đã có
from gradcam import analyze_image
from LLM_solution import get_treatment_solution
from inference import get_model

# CONFIG


CHECKPOINT_DIR = "weight/"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model 1 lần duy nhất khi khởi động app
print("Loading model...")
MODEL = get_model(CHECKPOINT_DIR)
print("Model ready!")

# HELPER
def format_top3(top3):
    lines = []
    medals = ["🥇", "🥈", "🥉"]
    for i, (name, conf) in enumerate(top3):
        crop, disease = name.split("___")
        disease_clean = disease.replace("_", " ")
        # Bỏ dấu backtick (thẻ code) đi, chỉ dùng in đậm để không bị nền đen
        lines.append(f"{medals[i]} **{conf:.1f}%** — {crop} ({disease_clean})")
    # Tăng khoảng cách giữa các dòng để dễ đọc hơn
    return "\n\n".join(lines)


def format_label(label, confidence):
    crop, disease = label.split("___")
    disease_clean = disease.replace("_", " ")
    is_healthy    = "healthy" in disease.lower()

    if is_healthy:
        status = "✅ Khỏe mạnh"
        color  = "🟢"
    elif confidence >= 80:
        color  = "🔴"
        status = "⚠️ Phát hiện bệnh"
    else:
        color  = "🟡"
        status = "⚠️ Có thể bị bệnh"

    return (
        f"## {color} {status}\n\n"
        f"**Loại cây:** {crop.replace('_', ' ')}\n\n"
        f"**Bệnh:** {disease_clean}\n\n"
        f"**Độ tin cậy:** {confidence:.1f}%"
    )



# PIPELINE CHÍNH


def analyze(image, use_llm):
    if image is None:
        yield None, "## ❌ Vui lòng tải ảnh lên", "", ""
        return

    try:
   
        yield None, "⏳ *Đang tính toán...*", "⏳ *Chờ chẩn đoán bệnh...*", "⏳ *Đợi giải pháp từ AI...*"

        #  1. ViT + EigenCAM 
        pil_image = Image.fromarray(image).convert("RGB")
        result    = analyze_image(pil_image, model=MODEL, checkpoint_dir=CHECKPOINT_DIR)

        label      = result["label"]
        confidence = result["confidence"]
        top3       = result["top3"]
        heatmap    = result["heatmap"]

        # Format kết quả
        diagnosis_md = format_label(label, confidence)
        top3_md      = f"### 📊 Top-3 dự đoán\n\n{format_top3(top3)}"

        # TRẢ KẾT QUẢ BƯỚC 1: Hiện ngay Tên Bệnh và Bản đồ nhiệt
        loading_msg = "### ⏳ AI đang soạn phác đồ điều trị...\n*Vui lòng đợi vài giây để nhận đơn thuốc chi tiết.*" if use_llm else "### ⏸️ Đã tắt trợ lý AI."
        yield heatmap, diagnosis_md, top3_md, loading_msg

        solution_md = ""
        if use_llm:
            crop, disease = label.split("___")
            crop_clean = crop.replace("_", " ")
            disease_clean = disease.replace("_", " ")
            full_disease_name = crop_clean + " " + disease_clean
            print(full_disease_name)
            if "healthy" in disease.lower():
                solution_md = "### 🌿 Cây khỏe mạnh — Không cần điều trị!"
            else:
                solution_text = get_treatment_solution(full_disease_name)
                solution_md   = f"### 💊 Giải pháp từ AI\n\n**BỆNH:** {full_disease_name}\n\n{solution_text}"

        # TRẢ KẾT QUẢ BƯỚC 2: Hoàn tất và hiện đơn thuốc AI
        yield heatmap, diagnosis_md, top3_md, solution_md

    except Exception as e:
        yield None, f"## ❌ Lỗi: {str(e)}", "", ""



custom_theme = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(
    title="Plant Disease Detection",
    theme=custom_theme,
    css="""
    /* Tùy chỉnh Header gradient */
    .custom-header { 
        text-align: center; 
        padding: 30px 20px; 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
        color: white; 
        border-radius: 16px; 
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        margin-bottom: 30px;
    }
    .custom-header h1 { margin: 0; font-size: 2.5em; font-weight: 800; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
    .custom-header p { margin-top: 10px; font-size: 1.1em; opacity: 0.95; font-weight: 500; }
    
    /* ÉP BUỘC CARD LUÔN SÁNG VÀ CHỮ ĐEN TRONG MỌI CHẾ ĐỘ */
    .result-card, .dark .result-card { 
        background-color: #ffffff !important; 
        color: #000000 !important; 
        border-radius: 12px; 
        padding: 20px; 
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Ép tất cả các thành phần text bên trong (p, h1, h2, li, strong) thành màu đen */
    .result-card *, .dark .result-card * {
        color: #000000 !important;
    }
    """
) as demo:

    # ── Header ──
    gr.HTML("""
    <div class="custom-header">
        <h1> AI Plant Doctor</h1>
        <p>Hệ thống Thị giác Máy tính chẩn đoán 38 loại bệnh cây trồng</p>
    </div>
    """)

    # ── Main layout ──
    with gr.Row():

        # CỘT TRÁI: Nhập liệu & Báo cáo thống kê
        with gr.Column(scale=5):
            gr.Markdown("### 📸 Phân tích hình ảnh")
            input_image = gr.Image(
                type="numpy",
                height=280,
                elem_classes="result-card"
            )
            
            # GOM CHUNG 1 HÀNG: Bật AI và Nút Phân tích
            with gr.Row():
                use_llm = gr.Checkbox(
                    label="🤖 Bật Trợ lý Nông nghiệp AI",
                    value=True,
                )
                analyze_btn = gr.Button(
                    "🔍 Phân Tích Ngay",
                    variant="primary",
                )

            # GOM CHUNG 1 HÀNG: Kết quả chẩn đoán và Top 3
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🩺 Kết quả", elem_classes="mt-4")
                    diagnosis_md = gr.Markdown(
                        value="*Chưa có dữ liệu.*",
                        elem_classes="result-card",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Top 3", elem_classes="mt-4")
                    top3_md = gr.Markdown(
                        value="*Chưa có thống kê.*",
                        elem_classes="result-card",
                    )

        # CỘT PHẢI: Giải pháp chuyên sâu & Hình ảnh phân tích
        with gr.Column(scale=6):
            gr.Markdown("### 💡 Đề xuất & Phân tích chuyên sâu")
            
            with gr.Tabs():
                with gr.TabItem("💊 Giải pháp từ AI"):
                    solution_md = gr.Markdown(
                        value="*Vui lòng phân tích ảnh để nhận phác đồ điều trị.*",
                        elem_classes="result-card"
                    )
                
                with gr.TabItem("🔥 Bản đồ Nhiệt (EigenCAM)"):
                    heatmap_output = gr.Image(
                        type="numpy",
                        height=400,
                        show_label=False
                    )

    # ── Accordion Hướng dẫn (Để ở dưới cùng) ──
    with gr.Accordion("📖 Hướng dẫn chụp ảnh chuẩn", open=False):
        gr.Markdown("""
        **Để AI dự đoán chính xác nhất, bạn cần chú ý:**
        * 🎯 **Trọng tâm:** Chụp rõ **1 lá bị bệnh**, để lá chiếm 70-80% khung hình.
        * ☀️ **Ánh sáng:** Chụp ở nơi đủ sáng, tránh bóng râm đổ trực tiếp lên lá.
        * 🌿 **Danh sách hỗ trợ:** Táo, Việt quất, Anh đào, Ngô (Bắp), Nho, Cam, Đào, Ớt chuông, Khoai tây, Mâm xôi, Đậu nành, Bí, Dâu tây, Cà chua.
        """)

    # ── Event ──
    analyze_btn.click(
        fn=analyze,
        inputs=[input_image, use_llm],
        outputs=[heatmap_output, diagnosis_md, top3_md, solution_md],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )