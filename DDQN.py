import random
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
import torch.nn.functional as F


memory_size = 500
batch_size = 32
GAMMA = 0.9


# 取gym的环境
env = gym.make('CartPole-v0')

# 神经网络部分
num_states = 4
num_actions = 2

# 主网
model = nn.Sequential()
model.add_module('fc1', nn.Linear(num_states, 32))
model.add_module('Relu2', nn.ReLU())
model.add_module('fc2', nn.Linear(32, num_actions))
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_def = nn.MSELoss()

# 固定网
model_2 = nn.Sequential()
model_2.add_module('fc1', nn.Linear(num_states, 32))
model_2.add_module('Relu2', nn.ReLU())
model_2.add_module('fc2', nn.Linear(32, num_actions))

model_2.load_state_dict(model.state_dict())

# 经验池部分
class exp:
    # 初始化经验池，定义起始index
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []
        self.index = 0
    # 将读取到的S,A,S_,R放入经验池中，用循环下标来完成满池出旧数据的功能,因为用的index而不是append，所以有append(None)
    def push(self, state, action, next_state, reward):
        self.in_builder = [state, action, next_state, reward]
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.index] = self.in_builder
        self.index = (self.index + 1) % self.memory_size
    # 取样函数
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

# DQN
class DQN:
    def __init__(self):
        pass
    def choose_actions(self, state, episode):
        # 伊普西落随着实验次数增加降低
        epsilon = 0.5 * (1 / (episode + 1))
        # if是利用，else是探索
        if epsilon < np.random.uniform(0, 1):
                model.eval()
                with torch.no_grad():
                    # 得到的state输入神经网络之前要转为tensor
                    state = torch.tensor(state, dtype=torch.float)
                    outputs = model(state)
                    outputs = outputs.unsqueeze(0)
                    action_index = torch.argmax(outputs, 1)
                    action = action_index.numpy()[0]
                    return action
        else:
            # 对于CartPole-v0来说动作就是左右，框架对应为0和1，这里用env.action_space.sample效果也一样
                action = env.action_space.sample()
                return action
    def learn(self):
        # 对经验池的数据取样
        sample_list = exp_1.sample(batch_size)
        sample_list = np.array(sample_list, dtype=object)

        state_list = []
        action_list = []
        next_state_list = []
        reward_list = []

        for x in sample_list:
            state_list.append(x[0])
            action_list.append(x[1])
            next_state_list.append(x[2])
            reward_list.append(x[3])

        state_list = np.array(state_list)
        next_state_list = np.array(next_state_list)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)


        state_list = torch.tensor(state_list, dtype=torch.float)
        next_state_list = torch.tensor(next_state_list, dtype=torch.float)
        action_list = torch.tensor(action_list, dtype=torch.int64).unsqueeze(1)
        reward_list = torch.tensor(reward_list, dtype=torch.float)

        Q_now = model(state_list).gather(1, action_list)
        Q_next_Max = model_2(next_state_list).max(1)[0]
        q_target = reward_list + GAMMA * Q_next_Max
        q_target = q_target.unsqueeze(1)

        model.train()
        optimizer.zero_grad()
        losses = loss_def(Q_now, q_target)
        losses.backward()
        optimizer.step()


exp_1 = exp(memory_size)
dqn = DQN()

for i in range(400):
    reward_number = 0
    s = env.reset()
    while True:
        env.render()
        a = dqn.choose_actions(s, i)
        # 做动作
        s_, _, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存入经验池
        exp_1.push(s, a, s_, r)
        reward_number += r
        if len(exp_1.memory) >= memory_size:
            if i % 2 == 0:
                model_2.load_state_dict(model.state_dict())
            dqn.learn()
        if done:
            print("回合数{},奖励总和{},经验池长度{}".format(i, reward_number, len(exp_1.memory)))
            break
        s = s_

