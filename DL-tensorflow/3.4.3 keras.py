import tensorflow as tf                # keras从tensorflow里调用
from tensorflow import keras

import numpy as np                              # 文章的四行导入，导入SGD要降级到v1
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizer_v1 import SGD

from keras import losses                        # losses函数要手动导入

tf.compat.v1.disable_v2_behavior()              # 将tensorflow2降级到1的配套部分
tf.compat.v1.disable_eager_execution()

np.random.seed(0)

model = Sequential([
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
])

model.compile(optimizer=SGD(lr=0.1), loss=losses.binary_crossentropy)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])         # 异或门的四种输入情况

Y = np.array([[0], [1], [1], [1]])

model.fit(X, Y, epochs=200, batch_size=1)

classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print('classified: ')
print(Y == classes)
print()
print('output probability: ')
print(prob)