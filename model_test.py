import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

# 给数据类别放置到列表数据中
CLASS_NAME = np.array(['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'])

# 设置图片大小
IMG_HEIGHT = 32
IM_WIDTH = 32

# 加载模型
model = load_model('model.h5')

src = cv2.imread("database/validation/PS/PS_59.bmp")
src = cv2.resize(src, (32, 32))
src = src.astype("int32")
src = src / 255

# 扩充数据的维度
test_img = tf.expand_dims(src, 0)
# print(test_img.shape)
# (1,32,32,3)    批次是1 通道是3 看起来是灰度的，但实际上是三通道的彩色图

# 预测
preds = model.predict(test_img)
# print(preds)
score = preds[0]
# print(score)
print('模型预测的结果为{}, 概率为{}'.format(CLASS_NAME[np.argmax(score)], np.max(score)))
