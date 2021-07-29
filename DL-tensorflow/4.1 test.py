from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from keras.layers import Dense, Activation
from keras import losses
from keras.optimizer_v1 import SGD
from keras.utils import np_utils
import numpy as np

tf.compat.v1.disable_v2_behavior()              # 将tensorflow2降级到1的配套部分
tf.compat.v1.disable_eager_execution()

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print('x_train:', x_train.shape, 'y_train:', y_train.shape, 'x_test:', x_test.shape, 'y_test:', y_test.shape)


#  挑选训练用的10000个数据
N_train = 10000
indices_tranin = np.random.permutation(range(60000))[:N_train]
x_train = x_train[indices_tranin]
y_train = y_train[indices_tranin]

#  挑选测试用的3000个数据
N_test = 3000
indices_test = np.random.permutation(range(10000))[:N_test]
x_test = x_test[indices_test]
y_test = y_test[indices_test]

#  输入转为10000 X 784， 输出转为10个结果的码
x_train = x_train.reshape(len(x_train), -1)
y_train = np_utils.to_categorical(y_train, 10)
x_test = x_test.reshape(len(x_test), -1)
y_test = np_utils.to_categorical(y_test, 10)

#  定义输入层，隐藏层，输出层
n_in = 784
n_hidden = 200
n_out = 10

#  定义模型
model = Sequential()

model.add(Dense(input_dim=n_in, units=n_hidden))
model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

#  训练方法定义，定义损失函数，优化方法和学习率，评估指标
model.compile(loss=losses.categorical_crossentropy, optimizer=SGD(lr=0.01), metrics=['accuracy'])

#  训练回合数，每次抓取数据数量
epochs = 1000
batch_size = 100
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

#  评估训练效果，evaluate输出损失和model.compile中定义的metrics
loss_and_metrics = model.evaluate(x_test, y_test)
print(loss_and_metrics)

