# 介绍
这里是《深度强化学习》的主要算法实现。为了方便阅读，单个算法的实现及调用放在一个文件中。调用方式简单：
```bash
mkdir -p output
python 07_dqn.py --do_train --output_dir output
```

离散环境：https://www.gymlibrary.ml/environments/classic_control/cart_pole/。
连续环境：


# 算法列表
| 章节                                  | 算法                         |
| ------------------------------------- | ---------------------------- |
| 7 DQN与Q学习                          | DQN                          |
| 8 SARSA算法                           | SARSA                        |
| 9 价值学习与高级技巧                  | Dueling DQN, Double DQN      |
| 10 策略梯度算法                       | REINFORCE, Actor Critic      |
| 11 带基线的策略梯度方法               | Advantage Actor Critic (A2C) |
| 12 策略学习高级技巧                   | TRPO                         |
| 13 连续控制                           | DDPG, TD3                    |
| 14 对状态的不完全观测                 |                              |
| 15 模仿学习                           | GAIL                         |
| 16 并行计算                           |                              |
| 17 多智能体系统                       |                              |
| 18 合作关系设定下的多智能体强化学习   |                              |
| 19 非合作关系设定下的多智能体强化学习 |                              |
| 20 注意力机制与多智能体强化学习       |                              |
| 21 AlphaGo 与蒙特卡洛树搜索           |                              |
| 22 现实世界中的应用                   |                              |