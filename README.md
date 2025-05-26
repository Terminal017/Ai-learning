# Ai-learing

收录关于学校 Ai 课程实验的内容

## Gridworld 强化学习项目

该项目实现了一个网格世界（Gridworld）环境下的强化学习算法，包括：

- 策略评估 (Policy Evaluation)
- 策略迭代 (Policy Iteration)
- 价值迭代 (Value Iteration)

### 使用方法

在项目根目录下运行：

```bash
python Gridworld/run_gridworld.py
```

### 文件结构

- `Gridworld/src/`: 源代码目录
  - `main.py`: 主程序逻辑
  - `policy.py`: 策略类和算法实现
  - `gridworld.py`: 网格世界环境
  - `map_parser.py`: 地图解析器
  - `policy_parser.py`: 策略解析器
- `Gridworld/data/`: 数据文件
  - `map01.grid`: 网格世界地图
  - `map01.policy`: 初始策略
