import os
import sys

# 将src目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(current_dir)

# 直接运行main模块中的代码
if __name__ == "__main__":
    from src.main import run_policy_evaluation, run_policy_iteration, run_value_iteration
    from src.map_parser import MapParser
    from src.policy_parser import PolicyParser
    from src.gridworld import GridWorld
    
    # 1. 加载地图
    map_parser = MapParser()
    grid_map = map_parser.parse_map("Gridworld/data/map01.grid")
    grid_world = GridWorld(grid_map)

    # 2. 加载初始策略
    policy_parser = PolicyParser()
    initial_policy = policy_parser.parse_policy("Gridworld/data/map01.policy")

    # 运行策略评估
    run_policy_evaluation(grid_world, initial_policy)

    # 运行策略迭代
    # 重新加载策略以获得干净的起点
    initial_policy_for_pi = policy_parser.parse_policy("Gridworld/data/map01.policy")
    run_policy_iteration(grid_world, initial_policy_for_pi)

    # 运行值迭代
    run_value_iteration(grid_world)

    print("\n所有算法演示完成。")
