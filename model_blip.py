# model_blip.py
"""
BLIP图像描述模型
直接使用预训练的BLIP，效果远好于自己组装的ViT+GPT2
"""
from transformers import BlipForConditionalGeneration, BlipProcessor


def build_blip_model():
    """
    加载预训练的BLIP模型

    BLIP架构:
    - 视觉编码器: ViT
    - 文本解码器: BERT
    - 专门为图像描述任务设计和训练
    """
    model_name = "Salesforce/blip-image-captioning-base"

    print(f"加载BLIP模型: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M")

    return model, processor


if __name__ == "__main__":
    model, processor = build_blip_model()
    print(f"\nProcessor类型: {type(processor)}")
    print(f"Model类型: {type(model)}")
    print("BLIP模型加载成功!")
