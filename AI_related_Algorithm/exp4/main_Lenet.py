import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.python import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 定义图像尺寸
IMG_SIZE = (200, 200)

# 定义数据路径
train_path = 'D:/chest_xray/train'
test_path = 'D:/chest_xray/test'
val_path = 'D:/chest_xray/val'

# 定义标签
labels = {'normal': 0, 'pneumonia': 1}

# 初始化数据和标签数组
data = []
labelss = []

i = 0
# 读取训练集图像
for lab in labels:
    label_path = os.path.join(train_path, lab)
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # 归一化
        data.append(img)
        labelss.append(labels[lab])
        i = i + 1

# 数据增强
# TODO: 在这里添加数据增强的代码，例如改变亮度、水平翻转、平移、变形等操作


# 将数据和标签转换为NumPy数组
train_data = np.array(data)
train_label = np.array(labelss)
# 定义数据增强器
datagen = ImageDataGenerator(rotation_range=20,  # 随机旋转角度范围
                             horizontal_flip=True,  # 随机水平翻转
                             width_shift_range=0.1,  # 随机水平平移
                             height_shift_range=0.1,  # 随机竖直平移
                             zoom_range=0.1)  # 随机缩放

# 对训练集数据进行数据增强
augmented_data = []
augmented_labels = []
for i in range(train_data.shape[0]):
    random_number = random.random()
    if random_number <= 0.4:
        img = train_data[i]
        label = train_label[i]
        augmented_data.append(img)
        augmented_labels.append(label)
        # 使用数据增强器生成新的样本
        augmented_img = datagen.random_transform(img)
        augmented_data.append(augmented_img)
        augmented_labels.append(label)

# 将增强后的数据和标签转换为NumPy数组
augmented_data = np.array(augmented_data)
augmented_labels = np.array(augmented_labels)

# 打印增强后的数据集形状
print("增强后的训练集数据形状:", augmented_data.shape)
# 读取测试集图像
test_data = []
test_labels = []
for label in labels:
    label_path = os.path.join(test_path, label)
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # 归一化
        test_data.append(img)
        test_labels.append(labels[label])

# 将测试集数据和标签转换为NumPy数组
test_data = np.array(test_data)
test_labels = np.array(test_labels)


# 读取测试集图像
val_data = []
val_labels = []
for label in labels:
    label_path = os.path.join(val_path, label)
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # 归一化
        val_data.append(img)
        val_labels.append(labels[label])

# 将测试集数据和标签转换为NumPy数组
val_data = np.array(val_data)
val_labels = np.array(val_labels)
# 打印数据集的形状
print("训练集数据形状:", train_data.shape)
print("测试集数据形状:", test_data.shape)



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(augmented_data, augmented_labels, batch_size=32, epochs=10, validation_data=(val_data, val_labels))


# 绘制训练过程中的准确率和损失曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 评估模型在测试集上的表现
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("测试集损失:", test_loss)
print("测试集准确率:", test_accuracy)
