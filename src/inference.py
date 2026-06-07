import os
import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor
from model import build_model, load_checkpoint, CLASS_NAMES

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "weight/"

_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
_model = None

def get_model(checkpoint_dir=CHECKPOINT_DIR):
    global _model
    if _model is None:
        _model = build_model().to(DEVICE)
        state = load_checkpoint(checkpoint_dir, _model, device=str(DEVICE), prefer="best")
        if state is None:
            raise RuntimeError(f"Không tìm thấy checkpoint trong: {checkpoint_dir}")
        _model.eval()
    return _model

def load_image(image_input):
    """Load image và return PIL + tensor (1, 3, 224, 224) - CÓ BATCH DIM"""
    if isinstance(image_input, str):
        pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil = Image.fromarray(image_input).convert("RGB")
    else:
        pil = image_input.convert("RGB")
    
    inputs = _processor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)  # (1, 3, 224, 224)
    
    return pil, pixel_values

@torch.no_grad()
def predict(image_input, model=None):
    if model is None:
        model = get_model()
    
    pil, pixel_values = load_image(image_input)  # pixel_values: (1, 3, 224, 224)
    
    outputs = model(pixel_values)
    probs   = torch.softmax(outputs.logits, dim=1)[0]

    top3_probs, top3_idxs = torch.topk(probs, k=3)
    top3 = [
        (CLASS_NAMES[idx.item()], round(prob.item() * 100, 2))
        for idx, prob in zip(top3_idxs, top3_probs)
    ]

    return {
        "label": CLASS_NAMES[top3_idxs[0].item()],
        "pre_idx": top3_idxs[0].item(),
        "confidence": round(top3_probs[0].item() * 100, 2),
        "top3": top3,
        "pil_image": pil,
        "pixel_values": pixel_values  # (1, 3, 224, 224)
    }
    
def predict_and_show(image_path, checkpoint_dir=CHECKPOINT_DIR):
    """Test nhanh: dự đoán + hiển thị ảnh với kết quả."""
    import matplotlib.pyplot as plt
    model  = get_model(checkpoint_dir)
    result = predict(image_path, model=model)
 
    label      = result["label"]
    confidence = result["confidence"]
    top3       = result["top3"]
 
    crop, disease = label.split("___")
    short_label   = disease.replace("_", " ")
 
    print("\n" + "=" * 45)
    print("✅ KẾT QUẢ CHẨN ĐOÁN:")
    print(f"   Loại cây : {crop}")
    print(f"   Bệnh     : {short_label}")
    print(f"   Tin cậy  : {confidence:.2f}%")
    print("\n   Top-3:")
    for name, conf in top3:
        c, d = name.split("___")
        print(f"     {conf:6.2f}%  {c} — {d.replace('_', ' ')}")
    print("=" * 45)
 
    color = "green" if confidence >= 80 else "red"
    plt.figure(figsize=(8, 6))
    plt.imshow(result["pil_image"])
    plt.axis("off")
    plt.title(
        f"Dự đoán: {short_label}\nĐộ tự tin: {confidence:.2f}%",
        fontsize=16, color=color, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    plt.show()