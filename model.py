# model.py
"""
图像描述模型
架构：ViT编码器 + 线性投影 + GPT2解码器
"""
import torch
import torch.nn as nn
from transformers import (
    ViTModel,          # 视觉编码器
    GPT2LMHeadModel,   # 文本解码器
    GPT2Tokenizer,
    ViTImageProcessor,
)


class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        vit_name="google/vit-base-patch16-224",
        gpt2_name="gpt2",
        freeze_vit=True,
        freeze_gpt2_partial=True,
    ):
        super().__init__()

        # ========== 1. 加载预训练的视觉编码器 ==========
        print(f"加载视觉编码器: {vit_name}")
        self.vit = ViTModel.from_pretrained(vit_name)
        vit_hidden_size = self.vit.config.hidden_size  # 768

        if freeze_vit:
            # 冻结ViT全部参数（我们只用它提取特征，不训练它）
            for param in self.vit.parameters():
                param.requires_grad = False
            print("  -> ViT参数已冻结")

        # ========== 2. 加载预训练的文本解码器 ==========
        print(f"加载文本解码器: {gpt2_name}")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        gpt2_hidden_size = self.gpt2.config.hidden_size  # 768

        # GPT2默认没有pad_token，需要处理
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id

        if freeze_gpt2_partial:
            # 冻结GPT2的大部分层，只训练最后几层
            # 这样可以节省显存，也防止过拟合
            for param in self.gpt2.parameters():
                param.requires_grad = False
            # 解冻最后2个Transformer层
            for layer in self.gpt2.transformer.h[-6:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # 解冻LM Head
            for param in self.gpt2.lm_head.parameters():
                param.requires_grad = True
            print("  -> GPT2大部分层已冻结，最后2层+LM Head可训练")

        # ========== 3. 图像-文本连接层（核心创新点之一）==========
        # 将ViT的输出映射到GPT2的维度
        # 虽然这里都是768，但实际中往往维度不同，所以投影层是必须的
        self.projection = nn.Sequential(
            nn.Linear(vit_hidden_size, gpt2_hidden_size),
            nn.LayerNorm(gpt2_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 可学习的图像token标记
        # 告诉GPT2"接下来的内容是图像信息"
        self.image_token_embed = nn.Parameter(
            torch.randn(1, 1, gpt2_hidden_size) * 0.02
        )

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params / 1e6:.1f}M")
        print(f"  可训练参数量: {trainable_params / 1e6:.1f}M")
        print(f"  训练比例: {trainable_params / total_params * 100:.1f}%")

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        前向传播

        参数:
            pixel_values: [batch, 3, 224, 224] 图像tensor
            input_ids: [batch, seq_len] 文本token ids
            attention_mask: [batch, seq_len] 文本attention mask
            labels: [batch, seq_len] 标签（用于计算loss）

        返回:
            loss (如果提供了labels)
            logits: [batch, 1+seq_len, vocab_size]
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # ---- Step 1: 图像编码 ----
        with torch.no_grad() if not any(p.requires_grad for p in self.vit.parameters()) else torch.enable_grad():
            vit_outputs = self.vit(pixel_values=pixel_values)
            # 取[CLS] token的输出作为图像的全局特征
            # shape: [batch, 768]
            image_features = vit_outputs.last_hidden_state[:, 0, :]

        # ---- Step 2: 投影到GPT2空间 ----
        # [batch, 768] -> [batch, 768]
        projected = self.projection(image_features)
        # 加上可学习的图像标记，然后扩展维度
        # [batch, 768] -> [batch, 1, 768]
        image_embeds = projected.unsqueeze(1) + self.image_token_embed

        # ---- Step 3: 文本编码 ----
        # [batch, seq_len] -> [batch, seq_len, 768]
        text_embeds = self.gpt2.transformer.wte(input_ids)

        # ---- Step 4: 拼接图像和文本 ----
        # 图像特征放在前面：[IMG] token1 token2 ... tokenN
        # [batch, 1+seq_len, 768]
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # ---- Step 5: 构建attention mask ----
        # 图像位置的mask全为1
        image_mask = torch.ones(
            batch_size, 1, dtype=attention_mask.dtype, device=device
        )
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)

        # ---- Step 6: 构建labels（如果有的话）----
        combined_labels = None
        if labels is not None:
            # 图像位置的label设为-100（PyTorch CrossEntropyLoss会忽略）
            image_labels = torch.full(
                (batch_size, 1), -100, dtype=labels.dtype, device=device
            )
            combined_labels = torch.cat([image_labels, labels], dim=1)

        # ---- Step 7: 输入GPT2 ----
        outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return outputs

    def generate_caption(self, pixel_values, max_length=50, num_beams=3):
        """
        推理时生成描述

        参数:
            pixel_values: [1, 3, 224, 224] 单张图片
            max_length: 最大生成长度
            num_beams: beam search宽度

        返回:
            caption: str 生成的描述文本
        """
        self.eval()
        with torch.no_grad():
            # 编码图像
            vit_outputs = self.vit(pixel_values=pixel_values)
            image_features = vit_outputs.last_hidden_state[:, 0, :]
            projected = self.projection(image_features)
            image_embeds = projected.unsqueeze(1) + self.image_token_embed

            # 从图像特征开始自回归生成
            # 使用GPT2的generate方法
            batch_size = pixel_values.shape[0]

            # 用image_embeds作为起始，逐步生成
            generated_ids = []
            current_embeds = image_embeds
            current_mask = torch.ones(
                batch_size, 1, dtype=torch.long, device=pixel_values.device
            )

            for step in range(max_length):
                outputs = self.gpt2(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                )
                # 取最后一个位置的logits
                next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
                next_token = torch.argmax(next_token_logits, dim=-1)  # [batch]
                generated_ids.append(next_token)

                # 检查是否生成了结束符
                if (next_token == self.tokenizer.eos_token_id).all():
                    break

                # 将新token编码并加入序列
                next_embeds = self.gpt2.transformer.wte(next_token).unsqueeze(1)
                current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(batch_size, 1, dtype=torch.long, device=pixel_values.device)
                ], dim=1)

            if generated_ids:
                generated_ids = torch.stack(generated_ids, dim=1)
                caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                caption = ""

            return caption


# ---- 测试代码 ----
if __name__ == "__main__":
    print("=" * 50)
    print("测试模型构建")
    print("=" * 50)

    model = ImageCaptionModel()

    # 创建假数据测试前向传播
    batch_size = 2
    fake_images = torch.randn(batch_size, 3, 224, 224)
    fake_ids = torch.randint(0, 50257, (batch_size, 32))
    fake_mask = torch.ones(batch_size, 32, dtype=torch.long)
    fake_labels = torch.randint(0, 50257, (batch_size, 32))

    print(f"\n输入:")
    print(f"  图像: {fake_images.shape}")
    print(f"  文本: {fake_ids.shape}")

    outputs = model(fake_images, fake_ids, fake_mask, labels=fake_labels)
    print(f"\n输出:")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"\n模型构建成功!")
