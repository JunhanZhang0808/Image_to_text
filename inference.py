# inference.py
"""
推理脚本：给一张图片，生成描述
"""
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, GPT2Tokenizer
from model import ImageCaptionModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(checkpoint_dir=None, device="cuda"):
    """加载训练好的模型"""
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints", "best_model")

    print(f"从 {checkpoint_dir} 加载模型...")

    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir)

    model = ImageCaptionModel(freeze_vit=True, freeze_gpt2_partial=True)

    weights_path = os.path.join(checkpoint_dir, "trainable_weights.bin")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("模型加载完成!")
    return model, tokenizer, image_processor


def caption_image(image_path, model, tokenizer, image_processor, device="cuda"):
    """为一张图片生成描述"""
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        caption = model.generate_caption(pixel_values, max_length=50)

    return caption


if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, image_processor = load_model(device=device)

    # 可以指定图片路径，或者默认用测试集里的一张
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # 默认用测试集第一张图
        test_dir = os.path.join(PROJECT_DIR, "data", "test_images")
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        test_image = os.path.join(test_dir, test_images[0])

    print(f"\n测试图片: {test_image}")
    caption = caption_image(test_image, model, tokenizer, image_processor, device)
    print(f"生成描述: {caption}")
