import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练集数据
data_train = './database/train/'
data_train = pathlib.Path(data_train)

# 读取验证集数据
data_val = './database/validation/'
data_val = pathlib.Path(data_val)

# 给数据类别放置到列表数据中
CLASS_NAME = np.array(['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'])

# 设置图片大小和批次数
BATCH_SIZE = 32
IMG_HEIGHT = 32
IM_WIDTH = 32

# 归一化数据  keras自带    之前使用sklearn的，因为那是表格数据
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# 训练集生成器
train_data_gen = image_generator.flow_from_directory(directory=str(data_train),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,  # 打乱数据
                                                     target_size=(IMG_HEIGHT, IM_WIDTH),  # 原本是200x200,压缩到32x32
                                                     classes=list(CLASS_NAME)
                                                     )
# 验证集生成器
val_data_gen = image_generator.flow_from_directory(directory=str(data_val),
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,  # 打乱数据
                                                   target_size=(IMG_HEIGHT, IM_WIDTH),  # 原本是200x200,压缩到32x32
                                                   classes=list(CLASS_NAME)
                                                   )

# 利用keras 搭建神经网络
model = keras.Sequential()   # 表示模型是 按层顺序堆叠的
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(32, 32, 3), activation='relu'))
# 第一层卷积层  卷积核的数量6，输出就是6个通道 尺寸5，    三通道的
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# 增加最大池化层    2x2  strides步长为2
model.add(Conv2D(filters=16, kernel_size=5,  activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=120, kernel_size=5,  activation='relu'))
# 平展层
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(6, activation='softmax'))  # 6个输出   分类任务

# 编译卷积神经网络
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# 传入数据进行训练
history = model.fit(train_data_gen, validation_data=val_data_gen, epochs=50)

# 保存训练好的模型
model.save('model.h5')

# 绘制loss值
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("CNN神经网络loss值")
plt.legend()
plt.show()

# 绘制准确度
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("CNN神经网络accuracy值")
plt.legend()
plt.show()

