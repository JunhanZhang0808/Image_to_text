# app.py
"""
部署脚本：Gradio网页界面
上传图片，AI生成描述
"""
import os
import gradio as gr
import torch
from PIL import Image
from transformers import AutoImageProcessor, GPT2Tokenizer
from model import ImageCaptionModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 加载模型（启动时只加载一次）==========
print("正在加载模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints", "best_model")

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


# ========== 推理函数 ==========
def generate_caption(image):
    if image is None:
        return "请上传一张图片"

    try:
        image = Image.fromarray(image).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            caption = model.generate_caption(pixel_values, max_length=50)

        return caption if caption else "模型未能生成描述"
    except Exception as e:
        return f"生成出错: {str(e)}"


# ========== 构建界面 ==========
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="numpy", label="上传图片"),
    outputs=gr.Textbox(label="生成的描述", lines=3),
    title="图像描述生成器",
    description="上传一张图片，AI会为你生成一段文字描述。",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
