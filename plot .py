import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import cycle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 创建保存图片的目录
os.makedirs('./picture/cm', exist_ok=True)
os.makedirs('./picture/roc', exist_ok=True)
os.makedirs('./picture/train_loss', exist_ok=True)

# 加载测试数据
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    './7153/test',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 1. 加载模型和权重
model = tf.keras.models.load_model('./weight/aug_xception_best_weights.h5')

# 2. 绘制混淆矩阵
print("Generating confusion matrix...")
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('./picture/cm/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 绘制ROC曲线
print("Generating ROC curves...")
n_classes = len(class_names)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('./picture/roc/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 绘制训练损失和准确率曲线
print("Generating training history plots...")
with open('./history/aug_train_history.pkl', 'rb') as f:
    history = pickle.load(f)

# 训练和验证准确率
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 训练和验证损失
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./picture/train_loss/training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots saved successfully!")