# train_blip.py
"""
BLIP微调训练脚本
在8G显存的4060上微调BLIP图像描述模型
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json

# ========== 配置 ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

CONFIG = {
    "batch_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_text_length": 64,
    "use_amp": True,
    "save_dir": os.path.join(PROJECT_DIR, "checkpoints_blip"),
    "log_every": 50,
    "save_every": 1,
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ========== 数据集 ==========
class BLIPCaptionDataset(Dataset):
    """BLIP专用数据集"""

    def __init__(self, image_dir, caption_file, processor, max_length=64):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        df = pd.read_csv(caption_file)

        self.samples = []
        for _, row in df.iterrows():
            image_name = str(row['image']).strip()
            caption = str(row['caption']).strip()
            image_path = os.path.join(image_dir, image_name)

            if os.path.exists(image_path) and len(caption) > 3:
                self.samples.append({
                    "image_path": image_path,
                    "caption": caption,
                })

        print(f"  加载 {len(self.samples)} 条数据")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        caption = sample["caption"]

        # BLIP的processor同时处理图像和文本
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def build_dataloaders(processor, batch_size=4):
    """构建DataLoader"""
    train_dataset = BLIPCaptionDataset(
        image_dir=os.path.join(DATA_DIR, "train_images"),
        caption_file=os.path.join(DATA_DIR, "train_captions.txt"),
        processor=processor,
    )

    val_dataset = BLIPCaptionDataset(
        image_dir=os.path.join(DATA_DIR, "val_images"),
        caption_file=os.path.join(DATA_DIR, "val_captions.txt"),
        processor=processor,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader


# ========== 训练 ==========
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    from model_blip import build_blip_model
    model, processor = build_blip_model()
    model = model.to(device)

    # 冻结视觉编码器，只训练文本解码器
    for param in model.vision_model.parameters():
        param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数: {trainable_count / 1e6:.1f}M / {total_count / 1e6:.1f}M ({trainable_count/total_count*100:.1f}%)")

    # 构建数据
    print("\n加载数据...")
    train_loader, val_loader = build_dataloaders(processor, CONFIG["batch_size"])
    print(f"训练batch数: {len(train_loader)}")
    print(f"验证batch数: {len(val_loader)}")

    # 优化器
    optimizer = torch.optim.AdamW(
        trainable_params, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )

    total_steps = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = GradScaler(enabled=CONFIG["use_amp"])

    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始训练BLIP!")
    print(f"Epochs: {CONFIG['num_epochs']}, 总步数: {total_steps}")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(enabled=CONFIG["use_amp"]):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()

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

        # 验证
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(enabled=CONFIG["use_amp"]):
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                val_loss += outputs.loss.item()
                val_steps += 1
                if val_steps >= 200:
                    break

        avg_val_loss = val_loss / val_steps
        print(f"验证loss: {avg_val_loss:.4f}")

        # 保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CONFIG["save_dir"], "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  -> 保存最佳模型 (val_loss={avg_val_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"训练完成! 最佳验证loss: {best_val_loss:.4f}")
    print(f"模型保存在: {CONFIG['save_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
