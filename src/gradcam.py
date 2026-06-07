import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
import os

from model import CLASS_NAMES
from inference import predict as predict_inference, get_model

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "weight/"

class ViTGradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def reshape_transform(tensor):
    result = tensor[:, 1:, :] 
    result = result.reshape(result.size(0), 14, 14, result.size(2)) 
    result = result.permute(0, 3, 1, 2)
    return result


def generate_gradcam(model, image_tensor, target_class_idx=None):
    """image_tensor: (1, 3, 224, 224) - CÓ BATCH DIM"""
    wrapped = ViTGradCAMWrapper(model)
    wrapped.eval()
    
    target_layers = [wrapped.model.vit.encoder.layer[-1].layernorm_before]
    targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None

    cam = EigenCAM(
        model=wrapped,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    input_tensor  = image_tensor.to(DEVICE)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return grayscale_cam

def overlay_heatmap(pil_image, grayscale_cam, threshold=0.3):
    """Blend heatmap với ảnh gốc"""
    if isinstance(pil_image, Image.Image):
        pil_image = pil_image.resize((224, 224))
    
    img_np = np.array(pil_image).astype(np.float32) / 255.0

    full_cam = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    full_cam = full_cam.astype(np.float32) / 255.0

    blend_factor = np.clip((grayscale_cam - threshold) / (1.0 - threshold), 0, 1)
    mask = np.stack([blend_factor, blend_factor, blend_factor], axis=-1)

    result = full_cam * mask + img_np * (1 - mask)

    return (result * 255).astype(np.uint8)

def analyze_image(image_path_or_pil, model=None, checkpoint_dir=CHECKPOINT_DIR):
    """Analyze ảnh: predict + generate heatmap"""
    if model is None:
        model = get_model(checkpoint_dir)

    result = predict_inference(image_path_or_pil, model=model)
    
    pil_image    = result["pil_image"]
    pred_idx     = result["pre_idx"]
    pixel_values = result["pixel_values"]  # (1, 3, 224, 224)
    
    grayscale_cam = generate_gradcam(model, pixel_values, target_class_idx=pred_idx)
    heatmap = overlay_heatmap(pil_image, grayscale_cam)

    return {
        "label":      result["label"],
        "confidence": result["confidence"],
        "top3":       result["top3"],
        "heatmap":    heatmap,
    }
