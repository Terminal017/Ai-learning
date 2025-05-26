import numpy as np
import matplotlib.pyplot as plt
from src.map_parser import MapParser, Map, Cell
from src.policy_parser import PolicyParser
from src.policy import Policy
from src.gridworld import GridWorld

# 添加中文字体支持
import matplotlib
import platform

# 根据操作系统配置中文字体
system = platform.system()
if system == 'Windows':
    matplotlib.rc('font', family='Microsoft YaHei')
elif system == 'Darwin':  # macOS
    matplotlib.rc('font', family='Arial Unicode MS')
elif system == 'Linux':
    matplotlib.rc('font', family='WenQuanYi Micro Hei')
    
# 设置全局字体属性
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def draw_value_function(V, grid_world, policy=None, title="Value Function"):
    fig, ax = plt.subplots(figsize=(grid_world.get_width(), grid_world.get_height()))
    
    # Reshape V to 2D for imshow
    V_reshaped = np.reshape(V, (grid_world.get_height(), grid_world.get_width()))
    
    # 修复：安全地计算最小值，处理空数组情况
    non_zero_values = V_reshaped[V_reshaped != 0]
    if non_zero_values.size > 0:  # 检查是否有非零值
        min_val = np.min(non_zero_values)
    else:
        min_val = 0  # 如果所有值都是0，设置最小值为0
    
    max_val = np.max(V_reshaped)
    
    # 确保最小值和最大值不同，以防止颜色映射问题
    if min_val == max_val:
        max_val = min_val + 1.0
    
    # Create a mask for walls to color them differently
    wall_mask = np.zeros_like(V_reshaped, dtype=bool)
    for cell in grid_world.get_cells():
        if cell.is_wall():
            row, col = cell.get_coords()
            wall_mask[row, col] = True
            V_reshaped[row, col] = 0 # Set wall values to 0 for imshow, will be overridden by mask

    # Create a colormap. You might need to adjust 'viridis' or use a custom one.
    cmap = plt.cm.viridis
    cmap.set_bad('black') # Set color for masked (wall) areas

    im = ax.imshow(V_reshaped, cmap=cmap)
    
    # Apply wall mask
    im.set_array(np.ma.masked_array(V_reshaped, mask=wall_mask))

    for cell in grid_world.get_cells():
        row, col = cell.get_coords()
        
        # Adjust text position for better centering if needed
        text_x = col
        text_y = row
        
        if cell.is_goal():
            text = ax.text(text_x, text_y, "X",
                           ha="center", va="center", color="white", fontsize=10, fontweight='bold')
        elif cell.is_wall():
            text = ax.text(text_x, text_y, "#",
                           ha="center", va="center", color="white", fontsize=10) # Walls often get '#'
        else:
            # Display value and policy action
            value = V[cell.get_index()]
            value_str = f"{value:.1f}" # Format value to 1 decimal place

            policy_char = ''
            if policy and policy.policy_action_for_cell(cell) == 'GO_NORTH':
                policy_char = '↑'
            elif policy and policy.policy_action_for_cell(cell) == 'GO_EAST':
                policy_char = '→'
            elif policy and policy.policy_action_for_cell(cell) == 'GO_SOUTH':
                policy_char = '↓'
            elif policy and policy.policy_action_for_cell(cell) == 'GO_WEST':
                policy_char = '←'
            
            # Combine value and policy character
            display_text = f"{value_str}\n{policy_char}" if policy_char else value_str
            
            text = ax.text(text_x, text_y, display_text,
                           ha="center", va="center", color="white", fontsize=8)

    ax.set_xticks(np.arange(grid_world.get_width()))
    ax.set_yticks(np.arange(grid_world.get_height()))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(-.5, grid_world.get_width(), 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_world.get_height(), 1), minor=True)
    ax.tick_params(which="minor", size=0)
    
    plt.title(title)
    plt.colorbar(im, ax=ax, label="State Value")
    plt.tight_layout()
    plt.show()


def run_policy_evaluation(grid_world, initial_policy):
    print("\n--- 运行策略评估 ---")
    evaluated_values = initial_policy.evaluate_policy(grid_world)
    # 使用英文标题替代中文，避免字体问题
    draw_value_function(evaluated_values, grid_world, initial_policy, "Policy Evaluation: Initial Policy Values")
    print("策略评估完成.")

def run_policy_iteration(grid_world, initial_policy):
    print("\n--- 运行策略迭代 ---")
    optimal_policy_pi = Policy.policy_iteration(initial_policy, grid_world)
    # 使用英文标题替代中文，避免字体问题
    draw_value_function(optimal_policy_pi.get_values(), grid_world, optimal_policy_pi, "Policy Iteration: Optimal Policy and Values")
    print("策略迭代完成.")

def run_value_iteration(grid_world):
    print("\n--- 运行值迭代 ---")
    optimal_policy_vi = Policy.value_iteration(grid_world)
    # 使用英文标题替代中文，避免字体问题
    draw_value_function(optimal_policy_vi.get_values(), grid_world, optimal_policy_vi, "Value Iteration: Optimal Policy and Values")
    print("值迭代完成.")

# 为了向后兼容
def main():
    # 1. 加载地图
    map_parser = MapParser()
    grid_map = map_parser.parse_map("../data/map01.grid")
    grid_world = GridWorld(grid_map)

    # 2. 加载初始策略
    policy_parser = PolicyParser()
    initial_policy = policy_parser.parse_policy("../data/map01.policy")

    # 运行策略评估
    run_policy_evaluation(grid_world, initial_policy)

    # 运行策略迭代
    initial_policy_for_pi = policy_parser.parse_policy("../data/map01.policy")
    run_policy_iteration(grid_world, initial_policy_for_pi)

    # 运行值迭代
    run_value_iteration(grid_world)

    print("\n所有算法演示完成.")

if __name__ == "__main__":
    main()