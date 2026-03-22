# explore_data.py
from datasets import load_dataset

dataset = load_dataset("jxie/flickr8k")

# 查看一条数据的完整结构
sample = dataset['train'][0]
for key, value in sample.items():
    if isinstance(value, str) and len(value) > 100:
        print(f"{key}: {value[:100]}...")
    else:
        print(f"{key}: {value}")

# 查看图片
from PIL import Image
import io

# 根据实际数据格式处理
if 'image' in sample:
    img = sample['image']
    print(f"图片类型: {type(img)}, 尺寸: {img.size}")
