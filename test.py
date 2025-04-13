import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import pickle

# 创建目录（如果不存在）
os.makedirs('./test_results', exist_ok=True)

# 设置测试集路径
test_folder_path = './3160'  # 替换为你的测试数据路径

# 数据预处理（与训练时一致）
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_folder_path,
    target_size=(299, 299),  # 必须与训练时尺寸一致
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # 测试时禁止打乱顺序
)


# 构建模型结构（必须与训练时完全一致）
def build_model():
    xception = tf.keras.applications.Xception(
        weights=None,  # 不加载ImageNet权重
        include_top=False,
        input_shape=(299, 299, 3))

    # 冻结前100层（与训练时一致）
    for layer in xception.layers[:100]:
        layer.trainable = False

    x = xception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='softmax')(x)  # 类别数需与训练时一致

    model = Model(inputs=xception.input, outputs=output)
    return model


# 加载模型和权重
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 加载预训练权重
weight_path = './weight/xception_original_best_weights.h5'
if os.path.exists(weight_path):
    model.load_weights(weight_path)
    print(f"成功加载权重: {weight_path}")
else:
    raise FileNotFoundError(f"权重文件不存在: {weight_path}")

# 评估模型
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n测试集评估结果:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

