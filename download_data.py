# download_data.py
from datasets import load_dataset

print("正在下载Flickr8k数据集...")
dataset = load_dataset("jxie/flickr8k")

print(f"训练集: {len(dataset['train'])} 条")
print(f"验证集: {len(dataset['validation'])} 条")
print(f"测试集: {len(dataset['test'])} 条")

# 查看完整结构
sample = dataset['train'][0]
print(f"\n样本keys: {list(sample.keys())}")

# 看看每条caption
for key in sorted(sample.keys()):
    if key == 'image':
        img = sample['image']
        print(f"image: 类型={type(img)}, 尺寸={img.size}")
    else:
        print(f"{key}: {sample[key][:80]}...")
