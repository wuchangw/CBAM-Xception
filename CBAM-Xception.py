import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Add, Multiply, Conv2D, Reshape, Lambda, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
import pickle

# 创建保存图片和权重的目录
if not os.path.exists('./weight'):
    os.makedirs('./weight')
if not os.path.exists('./history'):
    os.makedirs('./history')

# 设置训练集和测试集路径
train_folder_path = './7153/train'
val_folder_path = './7153/val'
test_folder_path = './7153/test'

# 数据预处理
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_folder_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_folder_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_folder_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 加载预训练的 Xception 模型
xception = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# 冻结前 100 层
for layer in xception.layers[:100]:
    layer.trainable = False


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

    # 更复杂的通道注意力融合策略：加权和
    channel_attention = Multiply()([avg_channel_attention, max_channel_attention])
    x_channel = Multiply()([input_tensor, channel_attention])

    # 空间注意力（Spatial Attention）
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x_channel)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x_channel)
    concat = Concatenate()([avg_pool, max_pool])

    spatial_attention = Conv2D(1, (3, 3), padding='same', activation='sigmoid', use_bias=False)(concat)

    # 加强空间注意力的表示能力，使用更小的卷积核
    spatial_attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid', use_bias=False)(spatial_attention)

    # 更复杂的空间注意力融合：加权和
    x = Multiply()([x_channel, spatial_attention])

    return x


# 在 Xception 基础上构建自定义模型
x = xception.output
x = squeeze_excite_block(x)  # 添加优化后的 CBAM 模块

# 全局平均池化层
x = GlobalAveragePooling2D()(x)

# 全连接层
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

# 输出层
output = Dense(4, activation='softmax')(x)

model = Model(inputs=xception.input, outputs=output)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 定义回调
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("./weight/7153_xception_best_weights.h5", monitor="val_accuracy", save_best_only=True,
                             save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_delta=0.0001, verbose=1)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=30,
    verbose=1,
    callbacks=[tensorboard, checkpoint, reduce_lr]
)

# 评估模型
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 保存训练历史
with open('./history/7153_train_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
