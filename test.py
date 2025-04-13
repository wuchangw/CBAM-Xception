import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Multiply, Conv2D, \
    Reshape, Lambda, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# 创建目录（如果不存在）
os.makedirs('E:/aa/test_results', exist_ok=True)

# 设置测试集路径
test_folder_path = 'E:/GMU/brain/3160' 
# 数据预处理
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_folder_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # 测试时不需要打乱数据
)


# 定义优化后的 CBAM 模块
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]

    # 通道注意力（Channel Attention）
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)

    avg_pool = Reshape((1, 1, filters))(avg_pool)
    max_pool = Reshape((1, 1, filters))(max_pool)

    avg_channel_attention = Dense(filters // ratio, activation='relu', use_bias=False)(avg_pool)
    max_channel_attention = Dense(filters // ratio, activation='relu', use_bias=False)(max_pool)

    avg_channel_attention = Dense(filters, activation='sigmoid', use_bias=False)(avg_channel_attention)
    max_channel_attention = Dense(filters, activation='sigmoid', use_bias=False)(max_channel_attention)

    channel_attention = Multiply()([avg_channel_attention, max_channel_attention])
    x_channel = Multiply()([input_tensor, channel_attention])

    # 空间注意力（Spatial Attention）
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x_channel)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x_channel)
    concat = Concatenate()([avg_pool, max_pool])

    spatial_attention = Conv2D(1, (3, 3), padding='same', activation='sigmoid', use_bias=False)(concat)
    spatial_attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid', use_bias=False)(spatial_attention)

    x = Multiply()([x_channel, spatial_attention])
    return x

xception = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=(299, 299, 3))

# 冻结前100层
for layer in xception.layers[:100]:
    layer.trainable = False

x = xception.output
x = squeeze_excite_block(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=xception.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# 加载预训练权重
weight_path = 'E:/GMU/brain/weight/aug_xception_best_weights.h5'
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

# 预测测试集
print("\n正在生成分类报告...")
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# 生成分类报告
report = classification_report(y_true, y_pred_classes, target_names=class_labels, digits=4)
print("\n分类报告:")
print(report)
