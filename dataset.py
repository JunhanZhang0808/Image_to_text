# dataset.py
"""
数据加载模块
从本地 data/ 文件夹读取图片和描述
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
import pandas as pd
import random

# ========== 项目根目录 ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")


class ImageCaptionDataset(Dataset):
    """
    从本地文件加载的图像描述数据集

    data目录结构:
    data/
    ├── train_images/      ← 图片文件夹
    │   ├── train_00000.jpg
    │   └── ...
    └── train_captions.txt ← 描述文件
        image,caption
        train_00000.jpg,"A dog is running"
    """

    def __init__(self, image_dir, caption_file, image_processor, tokenizer, max_length=64):
        """
        参数:
            image_dir: 图片文件夹路径，比如 data/train_images
            caption_file: 描述文件路径，比如 data/train_captions.txt
            image_processor: 图像预处理器
            tokenizer: 文本分词器
            max_length: 文本最大token数
        """
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取描述文件
        df = pd.read_csv(caption_file)
        print(f"  读取描述文件: {caption_file} ({len(df)} 行)")

        # 整理成列表
        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            image_name = str(row['image']).strip()
            caption = str(row['caption']).strip()
            image_path = os.path.join(image_dir, image_name)

            if os.path.exists(image_path) and len(caption) > 3:
                self.samples.append({
                    "image_path": image_path,
                    "caption": caption,
                })
            else:
                skipped += 1

        print(f"  有效样本: {len(self.samples)} 条")
        if skipped > 0:
            print(f"  跳过: {skipped} 条 (图片不存在或描述太短)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. 加载图像
        image = Image.open(sample["image_path"]).convert("RGB")
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # 2. 处理文本
        text_inputs = self.tokenizer(
            sample["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def build_dataloaders(tokenizer, image_processor, batch_size=4):
    """
    构建训练和验证的DataLoader
    直接从 data/ 文件夹读取
    """
    print("从本地文件构建数据集...")
    print(f"数据目录: {DATA_DIR}")

    train_dataset = ImageCaptionDataset(
        image_dir=os.path.join(DATA_DIR, "train_images"),
        caption_file=os.path.join(DATA_DIR, "train_captions.txt"),
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    val_dataset = ImageCaptionDataset(
        image_dir=os.path.join(DATA_DIR, "val_images"),
        caption_file=os.path.join(DATA_DIR, "val_captions.txt"),
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


# ========== 测试 ==========
if __name__ == "__main__":
    from transformers import AutoImageProcessor, GPT2Tokenizer

    print("=" * 50)
    print("测试本地数据加载")
    print("=" * 50)

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = build_dataloaders(tokenizer, processor, batch_size=2)

    print("\n取一个batch测试...")
    batch = next(iter(train_loader))

    print(f"\n一个batch的结构:")
    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    for i in range(batch['input_ids'].shape[0]):
        decoded = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
        print(f"\n  样本{i}: {decoded}")

    print("\n数据加载测试通过!")
