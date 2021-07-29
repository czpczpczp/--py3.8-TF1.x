import tensorflow as tf                # keras从tensorflow里调用
from tensorflow import keras
from keras import losses
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizer_v1 import SGD

tf.compat.v1.disable_v2_behavior()              # 将tensorflow2降级到1的配套部分
tf.compat.v1.disable_eager_execution()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

#  定义模型
model = Sequential()

#  输入层到隐藏层
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 隐藏层到输出层
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

#  训练
model.compile(loss=losses.binary_crossentropy, optimizer=SGD(lr=0.1))

#  喂数据
model.fit(X, Y, epochs=4000, batch_size=4)

classified = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

print("classified: ")
print(Y == classified)
print()
print('output probability: ')
print(prob)
