import os 
import torch
from transformers import ViTForImageClassification

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)                          
id2label    = {i: n for i, n in enumerate(CLASS_NAMES)}
label2id    = {n: i for i, n in enumerate(CLASS_NAMES)}


def build_model():
    model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            id2label=id2label,
            label2id=label2id,  
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
    
    print("Model built successfully.")
    
    return model

def load_checkpoint(checkpoint_dir, model, device = "cpu", prefer =  "last"):
    fname = "last_checkpoint.pt" if prefer == "last" else "best_checkpoint.pt"
    path = os.path.join(checkpoint_dir, fname)
    
    if not os.path.exists(path):
        print(f"Checkpoint file {path} does not exist.")
        return None
    
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Model loaded from checkpoint: {path}")
    return state