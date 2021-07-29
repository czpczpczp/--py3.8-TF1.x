from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from keras.layers import Dense, Activation
from keras import losses
from keras.optimizer_v1 import SGD
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()    # 把tensorflow2.0降级到1.0

# 定义数据部分
def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)   # 200个横坐标
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x)) # 使sinx偏移产生noise数据
    return sin(x) + noise

T = 100
f = toy_problem(T)  # t=0~200的数据集

# 准备数据部分,令可回溯的时间长度为25，那么就有201-25=176组数据集 ,因为要到f(T+1)
length_of_sequences = 2 * T
max_len = 25

data = []
target = []

# 生成176个数据，但是实际上最后用的是t=0~24的部分（这里的数据定义有大坑）
for i in range(0, length_of_sequences - max_len + 1):
    data.append(f[i:i + max_len])
    target.append(f[i + max_len])

# 数据变形, 其实是把一个很长的向量变成 N个输入组的多维?矩阵?
X = np.array(data).reshape(len(data), max_len, 1)
Y = np.array(target).reshape(len(target), 1)

# 训练数据和验证数据的分割
train_len = int(len(data) * 0.9)
test_len = len(data) - train_len

X_train = X[0:train_len]
X_test = X[train_len:train_len+test_len]
Y_train = Y[0:train_len]
Y_test = Y[train_len:train_len+test_len]

#  用tensorflow实现部分，51行之前都是在准备数据，这里仍然是定义模型，定义损失函数，定义训练方法就可以了

# 定义模型
def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
    # 定义权重
    def weight_variable(shape):
        initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    # 定义偏置
    def bias_variable(shape):
        initial = tf.compat.v1.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    # RNN的cell定义，initial_state要定义每批训练输入了多少组数据
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(n_hidden)
    initial_state = cell.zero_state(n_batch, tf.float32)

    # 继续定义隐藏层
    state = initial_state
    outputs = []   #  保存过去隐藏层的输出

    # 隐藏层依赖过去的输出，那么过去的变量和现在的变量要保持一致，不然变量完全不一样＝没训练，即x(t=1)和x(t=20)的参数得一致,简单理解就是虽然时间变了，但是用的还是那个模型，所以参数得一样
    with tf.compat.v1.variable_scope('RNN'):
        for t in range(maxlen):
            # t=0的时候是第一组数据还没有参数，t>0之后的数据都共享t=0的参数
            if t > 0:
                tf.compat.v1.get_variable_scope().reuse_variables()
            # 输入的是x(t=0~24)，单个x(t)是一个176乘输入维度1的数据,state是上一次隐藏层的输出h(t-1)，对于单层RNN，其实这里的cell_out和左边state都是h(t)，对于这个模型，输入维度只有1，也就是说RNN的输入维度取决于x[某，某，？]的？
            (cell_output, state) = cell(x[:, t, :], state)
            outputs.append(cell_output)

    # outputs[-1]代表了最后的cell_out,其实就是h(t=24)，输入到y里就是模型对t=25时刻的预测，t=25时刻的标准值在target[]里存放过了
    output = outputs[-1]
    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    y = tf.matmul(output, V) + c

    return y

# 定义损失函数
def loss(y, t):
    mse = tf.reduce_mean(tf.square(y - t))
    return mse

# 定义训练方法
def training(loss):
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999).minimize(loss)
    return train_step

# 模型参数以及一些训练过程
n_in = len(X[0][0])
n_hidden = 20
n_out = len(Y[0])

x = tf.compat.v1.placeholder(tf.float32, shape=[None, max_len, n_in]) # None代表每个t有多少个数据，max_len是有多少t，n_in才是具体输入维度
t = tf.compat.v1.placeholder(tf.float32, shape=[None, n_out])
n_batch = tf.compat.v1.placeholder(tf.int32, [])

y = inference(x, n_batch, maxlen=max_len, n_hidden=n_hidden, n_out=n_out)
loss = loss(y, t)
train_step = training(loss)

epochs = 500
batch_size = 10

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

history = []

n_t = train_len // batch_size

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)
    for i in range(n_t):
        start = i * batch_size
        end = start + batch_size
        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end],
            n_batch: batch_size
        })

    val_loss = sess.run(loss, feed_dict={
        x: X_test,
        t: Y_test,
        n_batch: 18
    })

    history.append(val_loss)
    print('epoch:', epoch,
          'loss', val_loss)

#  预测新的t的数据（这一部分理解的不好）
Z = X[:1]
origin = [f[i] for i in range(max_len)]
predicted = [None for i in range(max_len)]


for i in range(length_of_sequences - max_len + 1):
    z_ = Z[-1:]
    # 这里的1意思是比如x(1)，现在就只有一个数，上面的n_batch = 18意思是x(1)由18个数据构成
    y_ = sess.run(y, feed_dict={
        x: Z[-1:],
        n_batch: 1
    })
    # 第一个reshape是把y_变成下一次输入到模型中的序列的末尾,第二个是为了变成符合输入形式
    sequences_ = np.concatenate((z_.reshape(max_len, n_in)[1:], y_), axis=0).reshape(1, max_len, n_in)
    Z = np.append(Z, sequences_, axis=0)
    predicted.append(y_.reshape(-1))

plt.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
plt.rcParams['font.family'] = 'FangSong'

fig = plt.figure()

plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='#aaaaaa')
plt.plot(origin, linestyle='dashed', color='black')
plt.plot(predicted, color='black')

plt.show()
