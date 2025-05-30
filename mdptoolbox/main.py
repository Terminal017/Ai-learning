import mdptoolbox.example
import mdptoolbox.mdp
from gridworld_mdptoolbox import GridWorldMDPToolbox, create_complex_gridworld

def forest_example():
    """森林管理示例"""
    print("=" * 40)
    print("森林管理MDP示例")
    print("=" * 40)
    
    # 1. 定义 MDP (使用 forest 示例)
    # P: 转移概率矩阵
    # R: 奖励矩阵
    P, R = mdptoolbox.example.forest()

    # 2. 创建并运行价值迭代算法
    # discount=0.9 是折扣因子
    vi = mdptoolbox.mdp.ValueIteration(P, R, discount=0.9)
    vi.run()

    # 3. 打印最优策略
    print("价值迭代结果:")
    print(f"最优策略: {vi.policy}")
    print(f"价值函数: {vi.V}")
    print(f"迭代次数: {vi.iter}")

def gridworld_example():
    """Gridworld示例"""
    print("\n" + "=" * 40)
    print("GridWorld MDP示例")
    print("=" * 40)
    
    try:
        # 创建简单GridWorld MDP实例
        print("运行简单网格世界示例...")
        gridworld_mdp = GridWorldMDPToolbox()
        gridworld_mdp.compare_algorithms(discount=0.9)
        
        # 创建复杂GridWorld MDP实例
        print("\n运行复杂网格世界示例...")
        complex_gridworld_mdp = create_complex_gridworld()
        complex_gridworld_mdp.compare_algorithms(discount=0.9)
        
    except Exception as e:
        print(f"运行GridWorld示例时出错: {e}")

if __name__ == "__main__":
    # 运行森林示例
    forest_example()
    
    # 运行Gridworld示例
    gridworld_example()