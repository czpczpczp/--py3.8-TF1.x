from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from keras.layers import Dense, Activation
from keras import losses
from keras.optimizer_v1 import SGD
from keras.utils import np_utils
import numpy as np

from sklearn.utils import shuffle

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()    # 把tensorflow2.0降级到1.0

#定义模型,y = inference(x)
def inference(x, n_in, n_hidden, n_out):
    #  初始化权重
    def weight_variable(shape):
        initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    #  初始化偏置
    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)
    #  输入层-隐藏层，隐藏层-隐藏层
    for q, n_hiddens in enumerate(n_hidden):
        if q == 0:
            input = x
            input_dim = n_in
        else:
            input = h
            input_dim = n_hidden[q-1]

        W = weight_variable([input_dim, n_hiddens])
        b = bias_variable([n_hiddens])

        h = tf.nn.sigmoid(tf.matmul(input, W) + b)
        '''output = tf.nn.dropout(h, keep_prob)'''
    #  隐藏层到输出层
    W_out = weight_variable([n_hidden[-1], n_out])
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(h, W_out) + b_out)
    return y

#  定义损失函数
def loss(y, t):
    cross_entropy = tf.reduce_mean(-tf.compat.v1.reduce_sum(t * tf.compat.v1.log(y), reduction_indices=[1]))
    return cross_entropy


#  定义训练方法
def training(loss):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    return train_step


#  1.准备数据
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
#  2.设置模型
n_in = 784
n_hid = [400, 400]
n_out = 10

x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, n_in])
t = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, n_out])
'''keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32)'''

y = inference(x, n_in=n_in, n_hidden=n_hid, n_out=n_out)
loss = loss(y, t)
train_step = training(loss)
#  3.训练模型
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

batch_size = 100
n_batchsize = 100

for epoch in range(1000):
    X_, Y_ = shuffle(x_train, y_train)

    for i in range(n_batchsize):
        start = i * batch_size
        end = start + batch_size
        sess.run(train_step, feed_dict={
        x: X_[start:end],
        t: Y_[start:end]
        })
    if epoch % 25 == 0:
        print('epoch: ', epoch)
    #  4.评估模型
    '''correct_prediction = tf.reduce_mean(tf.compat.v1.to_float(tf.equal(tf.argmax(y, axis=1), tf.argmax(t, axis=1))))'''
curracy_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(curracy_prediction, tf.float32))
accu = sess.run(accuracy, feed_dict={
    x: x_test,
    t: y_test
})
print("accu:  ")
print(accu)