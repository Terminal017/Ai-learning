import numpy as np

from MDP import join


class MonteCarlo:
    # 采样序列
    @staticmethod
    def sample(MDP, Pi, timestep_max, number):
        S, A, P, R, gamma = MDP
        episodes = []
        for _ in range(number):
            episode = []
            timestep = 0
            s = S[np.random.randint(len(S) - 1)]  # 随机选择除一个终止状态外的状态作为起点
            # 一次采样 【到达终止状态或者到达最大时间步】
            while s != S[len(S) - 1] and timestep <= timestep_max:
                timestep += 1
                rand, temp = np.random.rand(), 0
                # 在状态 s 下根据策略选择动作
                for a_opt in A:
                    temp += Pi.get(join(s, a_opt), 0)
                    if temp > rand:
                        a = a_opt
                        r = R.get(join(s, a), 0)
                        break
                rand, temp = np.random.rand(), 0
                # 根据状态转移函数得到下一个状态
                for s_opt in S:
                    temp += P.get(join(join(s, a), s_opt), 0)
                    if temp > rand:
                        s_next = s_opt
                        break
                episode.append((s, a, r, s_next))
                s = s_next
            episodes.append(episode)
        return episodes

    # 计算价值
    @staticmethod
    def compute(episodes, V, N, gamma):
        for episode in episodes:
            G = 0
            # 从后往前计算
            for i in range(len(episode) - 1, -1, -1):
                (s, a, r, s_next) = episode[i]
                G = r + gamma * G
                N[s] += 1
                V[s] += (G - V[s]) / N[s]
