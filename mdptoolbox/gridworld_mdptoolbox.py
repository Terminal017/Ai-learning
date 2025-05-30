import numpy as np
import mdptoolbox
import mdptoolbox.mdp
import matplotlib.pyplot as plt
import matplotlib
import platform

# 配置中文字体
system = platform.system()
if system == 'Windows':
    matplotlib.rc('font', family='Microsoft YaHei')
elif system == 'Darwin':
    matplotlib.rc('font', family='Arial Unicode MS')
elif system == 'Linux':
    matplotlib.rc('font', family='WenQuanYi Micro Hei')
matplotlib.rcParams['axes.unicode_minus'] = False

class Cell:
    def __init__(self, row, col, cell_type, index):
        self.row = row
        self.col = col
        self.cell_type = cell_type
        self.index = index

    def is_wall(self):
        return self.cell_type == '#'

    def is_goal(self):
        return self.cell_type == 'X'

    def is_empty(self):
        return self.cell_type == ' '

    def get_coords(self):
        return (self.row, self.col)

    def get_index(self):
        return self.index

class GridWorldMDPToolbox:
    def __init__(self, grid_map_path=None, grid_string=None):
        """初始化GridWorld MDP环境"""
        # 加载地图
        if grid_map_path:
            self.grid_map = self._parse_map_file(grid_map_path)
        elif grid_string:
            self.grid_map = self._parse_map_string(grid_string)
        else:
            # 使用默认地图
            self.grid_map = self._create_default_map()
        
        # 定义动作
        self.actions = ['GO_NORTH', 'GO_EAST', 'GO_SOUTH', 'GO_WEST']
        self.num_actions = len(self.actions)
        self.num_states = len(self.grid_map['cells'])
        
        # 构建转移概率矩阵和奖励矩阵
        self.P, self.R = self._build_mdp_matrices()
    
    def _parse_map_file(self, file_path):
        """解析地图文件"""
        try:
            with open(file_path, 'r') as f:
                lines = [line.rstrip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"地图文件 {file_path} 未找到，使用默认地图")
            return self._create_default_map()
        
        return self._parse_map_lines(lines)
    
    def _parse_map_string(self, grid_string):
        """解析地图字符串"""
        lines = grid_string.strip().split('\n')
        return self._parse_map_lines(lines)
    
    def _parse_map_lines(self, lines):
        """解析地图行"""
        height = len(lines)
        width = len(lines[0]) if lines else 0
        cells = []
        index = 0
        
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                cells.append(Cell(r, c, char, index))
                index += 1
        
        return {
            'cells': cells,
            'width': width,
            'height': height
        }
    
    def _create_default_map(self):
        """创建默认的简单地图"""
        grid_string = """#####
#   #
# X #
#   #
#####"""
        return self._parse_map_string(grid_string)
    
    def get_cell_by_coords(self, row, col):
        """根据坐标获取格子"""
        if 0 <= row < self.grid_map['height'] and 0 <= col < self.grid_map['width']:
            return self.grid_map['cells'][row * self.grid_map['width'] + col]
        return None
    
    def propose_move(self, current_cell, action):
        """模拟动作效果"""
        if current_cell.is_goal():
            return current_cell
        
        new_row, new_col = current_cell.row, current_cell.col
        
        if action == 'GO_NORTH':
            new_row -= 1
        elif action == 'GO_EAST':
            new_col += 1
        elif action == 'GO_SOUTH':
            new_row += 1
        elif action == 'GO_WEST':
            new_col -= 1
        else:
            return current_cell
        
        proposed_cell = self.get_cell_by_coords(new_row, new_col)
        
        if proposed_cell is None or proposed_cell.is_wall():
            return current_cell
        else:
            return proposed_cell
    
    def _build_mdp_matrices(self):
        """构建MDP的转移概率矩阵P和奖励矩阵R"""
        P = np.zeros((self.num_actions, self.num_states, self.num_states))
        R = np.zeros((self.num_states, self.num_actions))
        
        cells = self.grid_map['cells']
        
        for action_idx, action in enumerate(self.actions):
            for state_idx, cell in enumerate(cells):
                if cell.is_wall():
                    # 墙壁状态：无法执行动作，留在原地
                    P[action_idx, state_idx, state_idx] = 1.0
                    R[state_idx, action_idx] = -10  # 墙壁惩罚
                    continue
                
                # 获取执行动作后的下一个状态
                next_cell = self.propose_move(cell, action)
                next_state_idx = next_cell.get_index()
                
                # 设置转移概率（确定性环境）
                P[action_idx, state_idx, next_state_idx] = 1.0
                
                # 设置奖励
                if next_cell.is_goal():
                    R[state_idx, action_idx] = 10  # 到达目标的奖励
                elif next_cell == cell:  # 撞墙或无效移动
                    R[state_idx, action_idx] = -1  # 轻微惩罚
                else:
                    R[state_idx, action_idx] = -0.1  # 移动成本
        
        return P, R
    
    def solve_value_iteration(self, discount=0.9, epsilon=0.01):
        """使用mdptoolbox的价值迭代算法"""
        print("开始价值迭代...")
        vi = mdptoolbox.mdp.ValueIteration(self.P, self.R, discount=discount, epsilon=epsilon)
        vi.run()
        
        print(f"价值迭代收敛，迭代次数: {vi.iter}")
        return vi.V, vi.policy
    
    def solve_policy_iteration(self, discount=0.9):
        """使用mdptoolbox的策略迭代算法"""
        print("开始策略迭代...")
        pi = mdptoolbox.mdp.PolicyIteration(self.P, self.R, discount=discount)
        pi.run()
        
        print(f"策略迭代收敛，迭代次数: {pi.iter}")
        return pi.V, pi.policy
    
    def solve_q_learning(self, discount=0.9, n_iter=10000):
        """使用mdptoolbox的Q学习算法"""
        print("开始Q学习...")
        ql = mdptoolbox.mdp.QLearning(self.P, self.R, discount=discount, n_iter=n_iter)
        ql.run()
        
        print(f"Q学习完成，迭代次数: {n_iter}")
        return ql.V, ql.policy
    
    def visualize_results(self, values, policy, title="MDP求解结果"):
        """可视化价值函数和策略"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 将一维数组重塑为2D网格
        height, width = self.grid_map['height'], self.grid_map['width']
        value_grid = np.reshape(values, (height, width))
        policy_grid = np.reshape(policy, (height, width))
        
        # 绘制价值函数
        im1 = ax1.imshow(value_grid, cmap='viridis')
        ax1.set_title(f"{title} - 价值函数")
        
        # 在每个格子上显示数值
        for i in range(height):
            for j in range(width):
                cell = self.get_cell_by_coords(i, j)
                if cell.is_wall():
                    text = "#"
                    color = "white"
                elif cell.is_goal():
                    text = "G"
                    color = "red"
                else:
                    text = f"{values[cell.get_index()]:.1f}"
                    color = "white"
                
                ax1.text(j, i, text, ha="center", va="center", 
                        color=color, fontsize=8, fontweight='bold')
        
        plt.colorbar(im1, ax=ax1)
        
        # 绘制策略
        im2 = ax2.imshow(np.zeros_like(policy_grid), cmap='gray')
        ax2.set_title(f"{title} - 最优策略")
        
        # 显示策略箭头
        action_chars = ['↑', '→', '↓', '←']
        for i in range(height):
            for j in range(width):
                cell = self.get_cell_by_coords(i, j)
                if cell.is_wall():
                    text = "#"
                    color = "white"
                elif cell.is_goal():
                    text = "G"
                    color = "red"
                else:
                    action_idx = policy[cell.get_index()]
                    text = action_chars[action_idx]
                    color = "blue"
                
                ax2.text(j, i, text, ha="center", va="center", 
                        color=color, fontsize=12, fontweight='bold')
        
        # 设置坐标轴
        for ax in [ax1, ax2]:
            ax.set_xticks(range(width))
            ax.set_yticks(range(height))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_algorithms(self, discount=0.9):
        """比较不同算法的结果"""
        print("=" * 50)
        print("使用MDPToolbox比较不同强化学习算法")
        print("=" * 50)
        
        # 价值迭代
        vi_values, vi_policy = self.solve_value_iteration(discount)
        self.visualize_results(vi_values, vi_policy, "价值迭代")
        
        # 策略迭代  
        pi_values, pi_policy = self.solve_policy_iteration(discount)
        self.visualize_results(pi_values, pi_policy, "策略迭代")
        
        # Q学习
        ql_values, ql_policy = self.solve_q_learning(discount, n_iter=1000)
        self.visualize_results(ql_values, ql_policy, "Q学习")
        
        # 比较结果
        print("\n算法比较:")
        print(f"价值迭代 - 最大价值: {np.max(vi_values):.3f}")
        print(f"策略迭代 - 最大价值: {np.max(pi_values):.3f}")
        print(f"Q学习 - 最大价值: {np.max(ql_values):.3f}")

def create_complex_gridworld():
    """创建一个复杂的网格世界示例"""
    complex_grid = """###################
#                X#
#   ###########   #
#   #         #   #
#   # ####### #   #
#   # #     # #   #
#     #  #  #     #
#        #        #
#                 #
###             ###
#                 #
###################"""
    
    return GridWorldMDPToolbox(grid_string=complex_grid)

def main():
    """主函数"""
    print("选择要运行的示例:")
    print("1. 简单网格世界")
    print("2. 复杂网格世界")
    print("3. 尝试加载外部地图文件")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        # 简单网格世界
        gridworld_mdp = GridWorldMDPToolbox()
        gridworld_mdp.compare_algorithms(discount=0.9)
        
    elif choice == "2":
        # 复杂网格世界
        gridworld_mdp = create_complex_gridworld()
        gridworld_mdp.compare_algorithms(discount=0.9)
        
    elif choice == "3":
        # 尝试加载外部文件
        try:
            gridworld_mdp = GridWorldMDPToolbox("../Gridworld/data/map01.grid")
        except:
            print("无法加载外部文件，使用默认地图")
            gridworld_mdp = GridWorldMDPToolbox()
        gridworld_mdp.compare_algorithms(discount=0.9)
        
    else:
        print("无效选择，使用默认简单网格世界")
        gridworld_mdp = GridWorldMDPToolbox()
        gridworld_mdp.compare_algorithms(discount=0.9)

if __name__ == "__main__":
    main()
