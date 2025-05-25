import numpy as np

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


def join(s1, s2):
    return s1 + '-' + s2


# 从字典中取出对应状态的参数数组
# 例如：从策略1中取出状态 S1 的动作概率分布，即[0.5,0.5]
def get_state_parameter(x, s):
    p = []
    for i in x.keys():
        if i.split('-')[0] == s:
            p.append(x[i])
    return p


# 计算a,b两个向量的内积
def compute_sum(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.inner(a, b)


# 将 MDP 的奖励函数转为 MRP 的奖励函数
# 思想：对于每个状态，根据策略将所有动作的概率进行加权，得到的奖励和就可以被认为是在 MRP 中该状态的奖励
def R_MDP_to_MRP(R, Pi, S):
    MRP_R = []
    for i in range(len(S)):
        MRP_R.append(compute_sum(get_state_parameter(Pi, S[i]), get_state_parameter(R, S[i])))
    return MRP_R


# 计算 MDP 的转移概率
def compute_P(P, Pi):
    P1 = P.copy()
    for i in P1.keys():
        P1[i] *= Pi[join(i.split('-')[0], i.split('-')[1])]
    return P1


# 根据转移概率创建 MRP 的转移矩阵
def set_MRP_P(P, S):
    MRP_P = np.zeros((len(S), len(S)))
    for i in P.keys():
        start_index = S.index(i.split('-')[0])
        end_index = S.index(i.split('-')[2])
        MRP_P[start_index][end_index] = P[i]
    MRP_P[len(S) - 1][len(S) - 1] = 1.0  # 终止状态设置转移概率
    return MRP_P


# 将 MDP 的转移函数转为 MRP 的转移矩阵
# 思想：对于每个状态转移到其他状态，计算策略的转移概率与状态转移函数的转移概率乘积和作为 MRP 的转移概率
def P_MDP_to_MRP(P, Pi, S):
    return set_MRP_P(compute_P(P, Pi), S)


# MDP = (0--S, 1--A, 2--P, 3--R, 4--gamma)
# 计算在状态 S 下采取动作 A 的价值 Q 
def compute_Q(s, a, MDP, V):
    r = MDP[3][join(s, a)]
    sum_PV = 0
    for i in range(len(MDP[0])):
        p = join(join(s, a), MDP[0][i])
        if p in MDP[2].keys():
            sum_PV += MDP[2][p] * V[i]

    return r + MDP[4] * sum_PV

# 计算状态动作（s, a）出现的频率，以此估计策略的占用度量
def occupancy(episodes, s, a, timestep_max, gamma):
    rho = 0
    total_times = np.zeros(timestep_max)
    occur_times = np.zeros(timestep_max)
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho
