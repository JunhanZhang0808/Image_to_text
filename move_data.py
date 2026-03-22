# move_data_to_project.py
"""
把HuggingFace缓存的数据搬到项目目录里
同时把图片全部导出成jpg文件，方便查看和管理
"""
import os
from datasets import load_dataset
from PIL import Image

# ========== 你的项目路径 ==========
PROJECT_DIR = r"C:\Users\curry\Desktop\mimo"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

# 创建文件夹
for d in [DATA_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 50)
print("把数据集搬到项目目录里")
print("=" * 50)
print(f"项目目录: {PROJECT_DIR}")
print(f"数据目录: {DATA_DIR}")
print()

# ========== 重新加载数据集 ==========
print("加载数据集...")
dataset = load_dataset("jxie/flickr8k")

# ========== 导出图片和描述 ==========
def export_split(split_data, split_name, img_dir):
    """
    导出一个split的所有图片和描述

    最终生成:
    - img_dir/ 下面是所有图片
    - data/{split_name}_captions.txt 是描述文件
    """
    caption_file = os.path.join(DATA_DIR, f"{split_name}_captions.txt")

    print(f"\n正在导出 {split_name} 集 ({len(split_data)} 张图片)...")

    with open(caption_file, "w", encoding="utf-8") as f:
        # 写表头
        f.write("image,caption\n")

        for i, item in enumerate(split_data):
            # 保存图片
            image = item['image']
            image_filename = f"{split_name}_{i:05d}.jpg"
            image_path = os.path.join(img_dir, image_filename)

            if not os.path.exists(image_path):
                image.save(image_path, quality=95)

            # 写描述
            for cap_idx in range(5):
                cap_key = f'caption_{cap_idx}'
                if cap_key in item:
                    caption = str(item[cap_key]).strip().replace('"', '""')
                    if len(caption) > 3:
                        f.write(f'{image_filename},"{caption}"\n')

            # 打印进度
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i + 1}/{len(split_data)}")

    print(f"  图片保存在: {img_dir}")
    print(f"  描述保存在: {caption_file}")

# 导出三个split
export_split(dataset['train'], "train", TRAIN_IMG_DIR)
export_split(dataset['validation'], "val", VAL_IMG_DIR)
export_split(dataset['test'], "test", TEST_IMG_DIR)

# ========== 验证导出结果 ==========
print("\n" + "=" * 50)
print("导出完成! 最终目录结构:")
print("=" * 50)

for root, dirs, files in os.walk(DATA_DIR):
    level = root.replace(DATA_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = '  ' * (level + 1)

    # 统计文件
    jpg_count = sum(1 for f in files if f.endswith('.jpg'))
    txt_count = sum(1 for f in files if f.endswith('.txt'))
    other_count = len(files) - jpg_count - txt_count

    if jpg_count > 0:
        print(f'{subindent}{jpg_count} 张jpg图片')
    if txt_count > 0:
        for f in files:
            if f.endswith('.txt'):
                print(f'{subindent}{f}')
    if other_count > 0:
        print(f'{subindent}其他文件 {other_count} 个')

print(f"\n你现在可以在文件管理器里打开:")
print(f"  {DATA_DIR}")
print(f"看到所有的图片和描述文件了!")
