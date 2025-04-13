import os
import cv2
import numpy as np
import random
from tqdm import tqdm  # 进度条（可选）

# 设置路径
source_dir = './7153/train/pituitary'
target_dir = './7153/train_aug/pituitary'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 支持的图像格式
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# 获取源目录中的所有图像文件
image_files = [f for f in os.listdir(source_dir)
               if os.path.splitext(f)[1].lower() in valid_extensions]

print(f"原始图像数量: {len(image_files)}")


# 定义数据增强函数
def augment_image(img):
    augmented_images = []

    # 1. 原始图像（直接保存）
    augmented_images.append(img)

    # 2. 随机缩放 (0.8~1.2倍)
    scale = random.uniform(0.8, 1.2)
    h, w = img.shape[:2]
    scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # 调整回原尺寸（避免尺寸不一致）
    if scale < 1.0:  # 缩小 → 填充黑边
        delta_w = w - scaled_img.shape[1]
        delta_h = h - scaled_img.shape[0]
        scaled_img = cv2.copyMakeBorder(scaled_img, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=0)
    else:  # 放大 → 裁剪中心部分
        start_x = (scaled_img.shape[1] - w) // 2
        start_y = (scaled_img.shape[0] - h) // 2
        scaled_img = scaled_img[start_y:start_y + h, start_x:start_x + w]
    augmented_images.append(scaled_img)

    # 3. 添加高斯噪声
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)  # 噪声强度15
    noisy_img = cv2.add(img, noise)
    augmented_images.append(noisy_img)

    # 4. 随机角度旋转 (-30°~30°)
    angle = random.uniform(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(rotated_img)

    return augmented_images


# 对每张图像进行增强并保存
for i, filename in enumerate(tqdm(image_files)):
    # 读取图像
    img_path = os.path.join(source_dir, filename)
    img = cv2.imread(img_path)

    # 进行数据增强
    augmented_images = augment_image(img)

    # 保存增强后的图像
    base_name = os.path.splitext(filename)[0]
    for j, aug_img in enumerate(augmented_images):
        if j == 0:  # 原始图像
            new_filename = f"{base_name}_orig.jpg"
        elif j == 1:  # 随机缩放
            new_filename = f"{base_name}_scaled.jpg"
        elif j == 2:  # 高斯噪声
            new_filename = f"{base_name}_noisy.jpg"
        else:  # 随机旋转
            new_filename = f"{base_name}_rot{random.randint(0, 360)}.jpg"

        save_path = os.path.join(target_dir, new_filename)
        cv2.imwrite(save_path, aug_img)

print(f"数据增强完成！增强后的图像数量: {len(image_files) * 4}")