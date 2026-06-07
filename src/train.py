import os
# from sched import scheduler 

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from model import build_model, load_checkpoint, CLASS_NAMES
from dataset import get_dataloader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
CONFIG = {
    "dataroot": "PlantVillage-Dataset-master/",
    "split_file": "PlantVillage-Dataset-master/splits",
    "checkpoint_dir": "weight",
    "data_type": "color",
    "plantdoc_root":  "PlantDoc/",
    "batch_size": 32,
    "plot_dir": "plots",
    "num_workers": 2,
    "label_smoothing": 0.1,
    "patience": 4,
    "phases": [
        {"phase":1, "epochs":3},
        {"phase":2, "epochs": 7},
        {"phase":3, "epochs": 15}
    ]
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _log(model, phase, note=""):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t = sum(p.numel() for p in model.parameters())
    print(f"\n[Phase {phase}] [{note}]  trainable: {n:,} / {t:,}  ({100*n/t:.1f}%)")
    
    
## Freeze/Unfreeze
def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True
    _log(model, phase = 1, note = "Backbone frozen, classifier unfrozen.")

def unfreeze_last_n_block(model, n: int =4):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True
    total =  len(model.vit.encoder.layer)
    
    for i in range(total-n,total):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = True
    _log(model, phase = 2, note = f"Last {n} blocks unfrozen.")
    
    
def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    _log(model, phase = 3, note = "All layers unfrozen.")

## Opimizer and Scheduler

def get_optimizer(model, phase: int):
    if phase == 1: 
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    elif phase == 2:
        total = len(model.vit.encoder.layer)
        n = 4
        groups = [{"params": list(model.vit.layernorm.parameters()), "lr": 3e-5}]
        for i in range(total-n, total):
            decay = 0.85 ** (total - 1 -i)
            groups.append({"params": list(model.vit.encoder.layer[i].parameters()), "lr": 1e-4 * decay})
        groups.append({"params": list(model.classifier.parameters()), "lr": 1e-4})
        return torch.optim.AdamW(groups, weight_decay=0.01)
    else:
        return torch.optim.AdamW([
            {"params": model.vit.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4}
        ],weight_decay=0.01)
        
        
## Save checkpoint
def save_checkpoint(checkpoint_dir, model, optimizer,scheduler, epoch, phase, val_f1, best_val_f1, is_best):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state ={
        "epoch": epoch,
        "phase": phase,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None
    }
    
    last_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    torch.save(state, last_path)
    print(f"Checkpoint saved: {last_path} epoch {epoch}, val_f1: {val_f1:.4f}")
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        torch.save(state, best_path)
        print(f"Best checkpoint updated: {best_path} epoch {epoch}, val_f1: {val_f1:.4f}")
        
 
def compute_metrics(all_labels, all_preds):
    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
           
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for image, label in tqdm(loader, desc=f"Epoch {epoch} [Train]"):
        image, label = image.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda"):
            outputs = model(image)
            loss = criterion(outputs.logits, label)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        all_preds.extend(outputs.logits.argmax(1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics
    
@torch.no_grad()
def validate(model, loader, criterion, epoch, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    
    for image, label in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
        image, label = image.to(DEVICE), label.to(DEVICE)
        
        with torch.autocast(device_type="cuda"):
            outputs = model(image)
            loss = criterion(outputs.logits, label)
            
        total_loss += loss.item()
        all_preds.extend(outputs.logits.argmax(1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())
     
    metrics = compute_metrics(all_labels, all_preds)
    if return_preds:
        return total_loss / len(loader), metrics, all_labels, all_preds
    return total_loss / len(loader), metrics

def plot_loss_curve(history, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1,len(history["train_loss"])+1)
    
    plt.figure(figsize=(9,5))
    plt.plot(epochs, history["train_loss"],"b-o", markersize=4 ,label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", markersize=4 , label="Val Loss")
    
    for ep in history["phase_boundaries"]:
        plt.axvline(x=ep, color="grey", linestyle="--",alpha=0.6)
        plt.text(ep+1, plt.ylim()[1]*0.95, f"Phase {history['phase_boundaries'].index(ep)+2}",fontsize=8, color="gray")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"),dpi=150)
    plt.close()

def plot_metrics_curve(history, plot_dir):

    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # Accuracy
    axes[0].plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    axes[0].plot(epochs, history["val_acc"],   "r-o", markersize=4, label="Val Acc")
    axes[0].set_title("Accuracy Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
 
    # F1-Score
    axes[1].plot(epochs, history["train_f1"], "b-o", markersize=4, label="Train F1")
    axes[1].plot(epochs, history["val_f1"],   "r-o", markersize=4, label="Val F1")
    axes[1].set_title("F1-Score Curve (Macro)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-Score")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
 
    # Phase boundaries
    for ep in history["phase_boundaries"]:
        for ax in axes:
            ax.axvline(x=ep, color="gray", linestyle="--", alpha=0.6)
 
    plt.tight_layout()
    path = os.path.join(plot_dir, "metrics_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Plot] Metrics curve → {path}")
 
 
def plot_confusion_matrix(all_labels, all_preds, plot_dir):
   
    os.makedirs(plot_dir, exist_ok=True)
 
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  
 
    short_names = [n.split("___")[-1].replace("_", " ")[:20] for n in CLASS_NAMES]
 
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized) — Test Set", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    path = os.path.join(plot_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    config = CONFIG
    print('Device:', DEVICE)
    
    # data
    train_loader, val_loader, test_loader = get_dataloader(
        dataroot = config["dataroot"],
        split_file = config["split_file"],
        data_type = config["data_type"],
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        plantdoc_root = config["plantdoc_root"]
    )
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}.")
    #model 
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scaler = GradScaler("cuda")
    
    resume = load_checkpoint(config["checkpoint_dir"], model, device=str(DEVICE))
    best_val_f1 = resume["best_val_f1"] if resume else 0.0
    start_phase = resume["phase"] if resume else 1
    start_epoch = resume["epoch"] if resume else 0
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
        "phase_boundaries": [],          
    }
 
    for phase_config in config["phases"]:
        phase = phase_config["phase"]
        epochs = phase_config["epochs"]
        
        if phase < start_phase:
            continue
        
        if phase == 1:
            freeze_backbone(model)
        elif phase == 2:
            unfreeze_last_n_block(model, n=4)
        else:
            unfreeze_all(model)
            
        optimizer = get_optimizer(model, phase)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        early_stop = 0  
        
        current_start_epoch = 1
        if resume and phase == start_phase:
            
            if "optimizer_state_dict" in resume and resume["optimizer_state_dict"]:
                optimizer.load_state_dict(resume["optimizer_state_dict"])
            if "scheduler_state_dict" in resume and resume["scheduler_state_dict"]:
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            
            # Chạy tiếp từ epoch bị ngắt + 1
            current_start_epoch = start_epoch + 1
            print(f" Resuming Phase {phase} from Epoch {current_start_epoch}...")
            
            # Nạp xong thì xóa biến resume để các Phase tiếp theo bắt đầu lại từ Epoch 1 bình thường
            resume = None
        

        for epoch in range(current_start_epoch, epochs + 1):
            tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
            val_loss, val_m = validate(model, val_loader, criterion, epoch)
            
            scheduler.step()
            
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_m["accuracy"])
            history["val_acc"].append(val_m["accuracy"])
            history["train_f1"].append(tr_m["f1"])
            history["val_f1"].append(val_m["f1"])
            
            print(f"\nPhase {phase} | Epoch {epoch}/{epochs}")
            print(f"  Train  loss={tr_loss:.4f}  acc={tr_m['accuracy']:.4f} r={tr_m['recall']:.4f} f1={tr_m['f1']:.4f}")
            print(f"  Val    loss={val_loss:.4f}  acc={val_m['accuracy']:.4f}  r={val_m['recall']:.4f} f1={val_m['f1']:.4f}")
            
            is_best = val_m['f1'] > best_val_f1
            
            if is_best:
                best_val_f1 = val_m['f1']
                early_stop = 0
            else:
                early_stop += 1
                
            save_checkpoint(
                checkpoint_dir = config["checkpoint_dir"],
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                epoch = epoch,
                phase = phase,  
                val_f1 = val_m['f1'],
                best_val_f1 = best_val_f1,
                is_best = is_best
            )
            
            if early_stop >= config["patience"]:
                print(f"Early stopping at epoch {epoch} in phase {phase} due to no improvement in val_f1 for {config['patience']} consecutive epochs.")
                break
            
        history["phase_boundaries"].append(len(history["train_loss"]))
        plot_loss_curve(history, config["plot_dir"])
        plot_metrics_curve(history, config["plot_dir"])
        
    print("Training completed. Evaluating on test set...")
    load_checkpoint(config["checkpoint_dir"], model, device=str(DEVICE), prefer="best")
    
    ts_loss, ts_m, ts_labels, ts_preds = validate(
    model, test_loader, criterion, epoch=0, return_preds=True)
    
    print(f"Test loss={ts_loss:.4f}  acc={ts_m['accuracy']:.4f}  f1={ts_m['f1']:.4f}")
    print(f"Best val F1: {best_val_f1:.4f}")
    
    plot_confusion_matrix(ts_labels, ts_preds, config["plot_dir"])
if __name__ == "__main__":
    main()