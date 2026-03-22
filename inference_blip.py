# inference_blip.py
"""
BLIP推理脚本（支持中英文）
"""
import os
import sys
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from translator import Translator

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(device="cuda"):
    checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints_blip", "best_model")
    print(f"从 {checkpoint_dir} 加载模型...")

    processor = BlipProcessor.from_pretrained(checkpoint_dir)
    model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.eval()

    # 加载翻译器
    translator = Translator()

    print("模型加载完成!")
    return model, processor, translator


def caption_image(image_path, model, processor, translator, device="cuda"):
    """生成中英文描述"""
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50, num_beams=3)

    english_caption = processor.decode(output[0], skip_special_tokens=True)
    chinese_caption = translator.translate(english_caption)

    return english_caption, chinese_caption


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, translator = load_model(device)

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_dir = os.path.join(PROJECT_DIR, "data", "test_images")
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        test_image = os.path.join(test_dir, test_images[0])

    print(f"\n测试图片: {test_image}")
    en, zh = caption_image(test_image, model, processor, translator, device)
    print(f"英文: {en}")
    print(f"中文: {zh}")
