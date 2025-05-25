from MDP import *
from MRP import compute
from MonteCarlo import MonteCarlo

S = ['S1', 'S2', 'S3', 'S4', 'S5']

A = ['保持S1', '前往S1', '前往S2', '前往S3', '前往S4', '前往S5', '概率前往']

P = {'S1-保持S1-S1': 1.0, 'S1-前往S2-S2': 1.0,
     'S2-前往S1-S1': 1.0, 'S2-前往S3-S3': 1.0,
     'S3-前往S4-S4': 1.0, 'S3-前往S5-S5': 1.0,
     'S4-前往S5-S5': 1.0, 'S4-概率前往-S2': 0.2, 'S4-概率前往-S3': 0.4, 'S4-概率前往-S4': 0.4}

R = {'S1-保持S1': -1, 'S1-前往S2': 0,
     'S2-前往S1': -1, 'S2-前往S3': -2,
     'S3-前往S4': -2, 'S3-前往S5': 0,
     'S4-前往S5': 10, 'S4-概率前往': 1}

gamma = 0.5

MDP = (S, A, P, R, gamma)

# 策略 1 ：随机策略
Pi_1 = {'S1-保持S1': 0.5, 'S1-前往S2': 0.5,
        'S2-前往S1': 0.5, 'S2-前往S3': 0.5,
        'S3-前往S4': 0.5, 'S3-前往S5': 0.5,
        'S4-前往S5': 0.5, 'S4-概率前往': 0.5}

# 策略 2 ：给定策略
Pi_2 = {'S1-保持S1': 0.6, 'S1-前往S2': 0.4,
        'S2-前往S1': 0.3, 'S2-前往S3': 0.7,
        'S3-前往S4': 0.5, 'S3-前往S5': 0.5,
        'S4-前往S5': 0.1, 'S4-概率前往': 0.9}


def MDP_to_MRP():
    (S, A, P, R, gamma) = MDP
    R = R_MDP_to_MRP(R, Pi_1, S)
    P = P_MDP_to_MRP(P, Pi_1, S)

    print('使用策略 1，将 MDP 转化为 MRP')
    print('转化后的 MRP 奖励函数：\n', R)
    print('\n转化后的 MRP 状态转移矩阵：\n', P)

    V = compute(P, R, gamma, len(S))
    print("\nMDP 中每个状态价值分别为\n", V)
    print("\n在状态为 S4 时采取动作 概率前往 的价值为：", compute_Q(S[3], A[6], MDP, V))


# MDP_to_MRP(MDP, Pi_1)

def MC():
    timestep_max = 20
    num = 1000
    V = {'S1': 0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0}
    N = {'S1': 0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0}
    episodes = MonteCarlo.sample(MDP, Pi_1, timestep_max, num)
    print("采样前 5 条序列为：\n")
    for i in range(5):
        if i >= len(episodes):
            break
        print("序列 %d：%s" % (i + 1, episodes[i]))
    MonteCarlo.compute(episodes, V, N, gamma)
    print('\n使用蒙特卡洛方法计算 MDP 的状态价值为\n', V)


# MC()


def occupancy_instance():
    gamma = 0.5
    timestep_max = 1000
    num = 1000
    s = S[3]
    a = A[6]
    episodes_1 = MonteCarlo.sample(MDP, Pi_1, timestep_max, num)
    episodes_2 = MonteCarlo.sample(MDP, Pi_2, timestep_max, num)
    rho_1 = occupancy(episodes_1, s, a, timestep_max, gamma)
    rho_2 = occupancy(episodes_2, s, a, timestep_max, gamma)
    print('策略1对状态动作对 (%s, %s) 的占用度量为：%f' % (s, a, rho_1))
    print('策略2对状态动作对 (%s, %s) 的占用度量为：%f' % (s, a, rho_2))


occupancy_instance()
