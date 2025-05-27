import mdptoolbox.example
import mdptoolbox.mdp

# 1. 定义 MDP (使用 forest 示例)
# P: 转移概率矩阵
# R: 奖励矩阵
P, R = mdptoolbox.example.forest()

# 2. 创建并运行价值迭代算法
# discount=0.9 是折扣因子
vi = mdptoolbox.mdp.ValueIteration(P, R, discount=0.9)
vi.run()

# 3. 打印最优策略
print(vi.policy)