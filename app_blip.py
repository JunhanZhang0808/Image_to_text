# app_blip.py
"""
BLIP网页部署（支持中英文）
"""
import os
import gradio as gr
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from translator import Translator

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

print("正在加载模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints_blip", "best_model")
processor = BlipProcessor.from_pretrained(checkpoint_dir)
model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir)
model = model.to(device)
model.eval()

translator = Translator()
print("所有模型加载完成!")


def generate_caption(image, language):
    if image is None:
        return "请上传一张图片"
    try:
        image = Image.fromarray(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=50, num_beams=3)

        english_caption = processor.decode(output[0], skip_special_tokens=True)

        if language == "中文":
            chinese_caption = translator.translate(english_caption)
            return chinese_caption
        elif language == "英文":
            return english_caption
        else:  # 中英双语
            chinese_caption = translator.translate(english_caption)
            return f"中文: {chinese_caption}\n\n英文: {english_caption}"

    except Exception as e:
        return f"出错: {str(e)}"


demo = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="numpy", label="上传图片"),
        gr.Radio(
            choices=["中文", "英文", "中英双语"],
            value="中文",
            label="输出语言"
        ),
    ],
    outputs=gr.Textbox(label="生成的描述", lines=4),
    title="图像描述生成器（支持中文）",
    description="上传一张图片，AI会为你生成文字描述",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
