import numpy as np

np.random.seed(0)

P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
P = np.array(P)
rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5


def compute_return(start, chain, gamma):
    G = 0
    for i in reversed(range(start, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G


def compute(P, rewards, gamma, states_num):
    rewards = np.array(rewards).reshape((-1, 1))  # 将 rewards 改写为列向量
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


def compute_return_sample():
    chain = [1, 2, 3, 6]
    start = 0
    G = compute_return(start, chain, gamma)
    print('计算回报为：%s' % G)


def compute_value_sample():
    V = compute(P, rewards, gamma, 6)
    print("马尔可夫奖励过程中每个状态的价值分别为：\n", V)


compute_value_sample()

