# train.py
"""
训练脚本
在8G显存的4060上训练图像描述模型
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json

from model import ImageCaptionModel
from dataset import build_dataloaders          # ← 已经改好了，直接用
from transformers import AutoImageProcessor, GPT2Tokenizer


def train():
    # ==================== 配置 ====================
    CONFIG = {
        # 数据
        "batch_size": 4,
        "max_text_length": 64,

        # 模型
        "vit_name": "google/vit-base-patch16-224",
        "gpt2_name": "gpt2",
        "freeze_vit": True,
        "freeze_gpt2_partial": True,

        # 训练
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "num_epochs": 15,
        "warmup_steps": 200,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "use_amp": True,

        # 保存
        "save_dir": "./checkpoints",
        "log_every": 50,
        "save_every": 1,
    }

    # 项目根目录
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG["save_dir"] = os.path.join(PROJECT_DIR, "checkpoints")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    with open(os.path.join(CONFIG["save_dir"], "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ==================== 设备 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ==================== 加载组件 ====================
    print("\n加载tokenizer和image processor...")
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG["gpt2_name"])
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(CONFIG["vit_name"])

    # ==================== 构建数据 ====================
    print("\n构建数据集...")
    train_loader, val_loader = build_dataloaders(
        tokenizer, image_processor, batch_size=CONFIG["batch_size"]
    )
    print(f"训练batch数: {len(train_loader)}")
    print(f"验证batch数: {len(val_loader)}")

    # ==================== 构建模型 ====================
    print("\n构建模型...")
    model = ImageCaptionModel(
        vit_name=CONFIG["vit_name"],
        gpt2_name=CONFIG["gpt2_name"],
        freeze_vit=CONFIG["freeze_vit"],
        freeze_gpt2_partial=CONFIG["freeze_gpt2_partial"],
    )
    model = model.to(device)

    # ==================== 优化器 ====================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n可训练参数: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )

    total_steps = len(train_loader) * CONFIG["num_epochs"] // CONFIG["gradient_accumulation_steps"]

    def lr_lambda(current_step):
        if current_step < CONFIG["warmup_steps"]:
            return float(current_step) / float(max(1, CONFIG["warmup_steps"]))
        progress = float(current_step - CONFIG["warmup_steps"]) / float(max(1, total_steps - CONFIG["warmup_steps"]))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=CONFIG["use_amp"])

    # ==================== 训练循环 ====================
    print(f"\n{'='*60}")
    print(f"开始训练!")
    print(f"总epochs: {CONFIG['num_epochs']}")
    print(f"总训练步数: {total_steps}")
    print(f"等效batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"{'='*60}\n")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            with autocast(enabled=CONFIG["use_amp"]):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / CONFIG["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * CONFIG["gradient_accumulation_steps"]

            if (step + 1) % CONFIG["log_every"] == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "gpu": f"{gpu_mem:.1f}G"
                })

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} 完成. 平均loss: {avg_epoch_loss:.4f}")

        # 保存模型
        if (epoch + 1) % CONFIG["save_every"] == 0:
            val_loss = evaluate(model, val_loader, device, CONFIG)
            print(f"验证loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(CONFIG["save_dir"], "best_model")
                save_model(model, tokenizer, image_processor, save_path)
                print(f"  -> 保存最佳模型")

            save_path = os.path.join(CONFIG["save_dir"], f"epoch_{epoch+1}")
            save_model(model, tokenizer, image_processor, save_path)

    print(f"\n训练完成! 最佳验证loss: {best_val_loss:.4f}")


def evaluate(model, val_loader, device, config):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            with autocast(enabled=config["use_amp"]):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            num_batches += 1
            if num_batches >= 200:
                break

    model.train()
    return total_loss / num_batches


def save_model(model, tokenizer, image_processor, save_path):
    os.makedirs(save_path, exist_ok=True)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if "projection" in k or "image_token" in k or "gpt2.transformer.h.10" in k or "gpt2.transformer.h.11" in k or "lm_head" in k
    }
    torch.save(trainable_state, os.path.join(save_path, "trainable_weights.bin"))
    tokenizer.save_pretrained(save_path)
    image_processor.save_pretrained(save_path)
    print(f"  模型已保存到: {save_path}")


if __name__ == "__main__":
    train()
