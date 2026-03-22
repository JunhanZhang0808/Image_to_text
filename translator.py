# translator.py
"""
英文翻译成中文
使用Helsinki-NLP的翻译模型，完全本地运行，免费
"""
from transformers import pipeline
import torch


class Translator:
    def __init__(self):
        print("加载翻译模型...")
        device = 0 if torch.cuda.is_available() else -1
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-zh",
            device=device,
        )
        print("翻译模型加载完成!")

    def translate(self, english_text):
        """英文翻译成中文"""
        result = self.translator(english_text, max_length=100)
        return result[0]['translation_text']


# 测试
if __name__ == "__main__":
    t = Translator()

    test_sentences = [
        "A brown dog is running through the grass.",
        "A man wearing red and black is standing on a bicycle.",
        "Two children are playing on the beach.",
    ]

    for s in test_sentences:
        chinese = t.translate(s)
        print(f"英文: {s}")
        print(f"中文: {chinese}")
        print()
