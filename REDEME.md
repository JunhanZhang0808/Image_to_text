# 图像描述生成器

上传一张图片，AI 自动生成中英文描述

## 项目结构

mimo/
├── data/ # 数据集（图片+描述）
├── checkpoints/ # ViT+GPT2 模型权重
├── checkpoints_blip/ # BLIP 模型权重
├── model.py # ViT+GPT2 模型
├── model_blip.py # BLIP 模型
├── dataset.py # 数据加载
├── train.py # 训练（ViT+GPT2）
├── train_blip.py # 训练（BLIP，推荐）
├── translator.py # 英译中
├── inference.py # 推理（ViT+GPT2）
├── inference_blip.py # 推理（BLIP+中文）
├── app.py # 网页部署（ViT+GPT2）
└── app_blip.py # 网页部署（BLIP+中文）


## 环境安装

```bash
conda create -n caption python=3.10 -y
conda activate caption
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft pillow gradio pandas tqdm sentencepiece

# 使用方法

# 1. 验证环境
python verify.py

# 2. 训练（推荐BLIP方案）
python train_blip.py

# 3. 推理
python inference_blip.py data/test_images/test_00005.jpg

# 4. 启动网页
python app_blip.py

技术栈

组件	技术

框架	PyTorch

模型	BLIP / ViT + GPT2

翻译	Helsinki-NLP opus-mt-en-zh

部署	Gradio

数据集	Flickr8k（8000张图片）

常见问题

显存不够：把 batch_size 从 4 改成 2

symlink 警告：忽略即可

验证只跑到 16%：正常，只跑 200 个 batch 节省时间