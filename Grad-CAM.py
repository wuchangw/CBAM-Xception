import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Multiply, Conv2D, Reshape, Lambda, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import pickle
# 创建目录（如果不存在）
# 设置测试集路径
test_folder_path = ./dataset' 
# 数据预处理
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_folder_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
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

# 构建模型结构（必须与保存权重时的结构完全一致）
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

# 加载模型权重
model.load_weights('./weight/aug_xception_best_weights.h5')#权重
model.summary()

# 加载训练历史并绘图
with open('./history/7153_train_history.pkl', 'rb') as f:#训练历史
    saved_history = pickle.load(f)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names, pred_index=None):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 记录操作的梯度
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 对于输出特征图的梯度
    grads = tape.gradient(class_channel, conv_outputs)

    # 全局平均池化来获得卷积输出特征图的梯度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 权重特征图
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU激活并归一化
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.1):  # 增加alpha值以增强热图颜色的强度
    # 用于显示
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # 归一化热图
    heatmap = np.uint8(255 * heatmap)

    # 使用jet颜色映射
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # 调整热图强度
    jet_heatmap = jet_heatmap * alpha  # 增加此处的alpha值可以增强颜色强度

    # 叠加热图和原图
    superimposed_img = jet_heatmap + img  # 移除乘以alpha，直接添加增强的热图
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # 显示图像
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

img_path = './data/glioma_tumor/G_1.jpg'#图片地址
img = load_img(img_path)
img_array = img_to_array(img)
plt.imshow(img_array.astype('uint8'))  # 将图像数据类型转换为无符号整型
plt.axis('off')  # 不显示坐标轴
plt.show()
img = load_img(img_path, target_size=(299,299))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
img_array /= 255.0  # 归一化
last_conv_layer_name = 'conv2d_11'# 确定最后一个卷积层的名称
# 应用Grad-CAM
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, ['dense_1', 'dropout', 'dense_2'])
display_gradcam(img_path, heatmap)


