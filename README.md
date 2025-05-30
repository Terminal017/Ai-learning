# Ai-learing

收录关于学校 Ai 课程实验的内容

## MDPToolbox 示例

使用 Python 的 mdptoolbox 库实现马尔可夫决策过程(MDP)算法，包括：

- 森林管理示例
- Gridworld 强化学习实现

### 使用方法

```bash
# 安装依赖
pip install mdptoolbox numpy matplotlib

# 运行示例
python mdptoolbox/main.py

# 只运行Gridworld示例
python mdptoolbox/gridworld_mdptoolbox.py
```

## Gridworld 强化学习项目

该项目实现了一个网格世界（Gridworld）环境下的强化学习算法，包括：

### 原生实现版本

- 策略评估 (Policy Evaluation)
- 策略迭代 (Policy Iteration)
- 价值迭代 (Value Iteration)

### MDPToolbox 实现版本

- 价值迭代 (Value Iteration)
- 策略迭代 (Policy Iteration)
- Q 学习 (Q-Learning)

### 使用方法

```bash
# 运行原生实现
python Gridworld/run_gridworld.py

# 运行MDPToolbox实现
python mdptoolbox/gridworld_mdptoolbox.py
```

### 文件结构

```
├── mdptoolbox/                    # MDPToolbox实现
│   ├── main.py                   # 主程序（包含森林和Gridworld示例）
│   └── gridworld_mdptoolbox.py   # Gridworld的MDPToolbox实现
├── Gridworld/                    # 原生实现
│   ├── src/                      # 源代码目录
│   │   ├── main.py              # 主程序逻辑
│   │   ├── policy.py            # 策略类和算法实现
│   │   ├── gridworld.py         # 网格世界环境
│   │   ├── map_parser.py        # 地图解析器
│   │   └── policy_parser.py     # 策略解析器
│   ├── data/                    # 数据文件
│   │   ├── map01.grid          # 网格世界地图
│   │   └── map01.policy        # 初始策略
│   └── run_gridworld.py        # 运行脚本
```

### 算法对比

- **原生实现**: 更详细的算法过程展示，适合学习理解
- **MDPToolbox 实现**: 使用成熟库，更高效稳定，支持更多算法

### 依赖项

```bash
pip install numpy matplotlib pymdptoolbox
```
