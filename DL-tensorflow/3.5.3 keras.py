import tensorflow as tf                # keras从tensorflow里调用
from tensorflow import keras
from keras import losses
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizer_v1 import SGD
from sklearn.utils import shuffle

tf.compat.v1.disable_v2_behavior()              # 将tensorflow2降级到1的配套部分
tf.compat.v1.disable_eager_execution()

M = 2  # 输入数据维度
K = 3  # 分类数量
n = 100  # 每个分类的数据量
N = n * K  # 全部数据个数

#  生成数据
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])

Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)   # X和Y都是向下做联接，Y每个都循环一百次是为了和X对应的分类tn对应上
Y = np.concatenate((Y1, Y2, Y3), axis=0)

#  keras部分
model = Sequential([
    Dense(input_dim=M, units=K),
    Activation('softmax')
])

model.compile(loss=losses.categorical_crossentropy, optimizer=SGD(lr=0.1))

minibatch_size = 50
model.fit(X, Y, epochs=20, batch_size=minibatch_size)
X_, Y_ = shuffle(X, Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=1)

print('classified: ')
#  model.predict返回的是概率，用np.argmax找索引
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)
print()
print('output probability: ')
print(prob)
