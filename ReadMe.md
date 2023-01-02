# 介绍
这里是《深度强化学习》的主要算法实现。为了方便阅读，单个算法的实现及调用放在一个文件中。调用方式简单：
```bash
mkdir -p output
python -u 04_dqn.py --do_train --output_dir output 2>&1 | tee output/log.txt
```


# 环境
根据动作状态空间是否连续，我们考虑两种环境：
- 离散环境：CartPole，https://www.gymlibrary.dev/environments/classic_control/cart_pole/.
- 连续环境：Pendulum，https://www.gymlibrary.dev/environments/classic_control/pendulum/.

测试环境是python3.7，依赖安装：
```bash
pip install -r requirements.txt 
```

所有代码均用于教学，可在笔记本CPU环境下训练。

# 算法列表
| 章节                                  | 算法                                                                                                             |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1 机器学习基础                        | MNIST                                                                                                            |
| 2 蒙特卡洛                            | PI approximation                                                                                                 |
| 3 强化学习基本概念                    | CartPole                                                                                                         |
| 4 DQN与Q学习                          | DQN                                                                                                              |
| 5 SARSA算法                           | SARSA                                                                                                            |
| 6 价值学习与高级技巧                  | Dueling DQN, Double DQN                                                                                          |
| 7 策略梯度算法                        | REINFORCE, Actor Critic                                                                                          |
| 8 带基线的策略梯度方法                | REINFORCE with baseline, A2C                                                                                     |
| 9 策略学习高级技巧                    | TRPO                                                                                                             |
| 10 连续控制                           | DDPG, TD3                                                                                                        |
| 11 对状态的不完全观测                 |                                                                                                                  |
| 12 模仿学习                           | GAIL                                                                                                             |
| 13 并行计算                           | A3C                                                                                                              |
| 14 多智能体系统                       | MPE                                                                                                              |
| 15 合作关系设定下的多智能体强化学习   | MAC-A2C                                                                                                          |
| 16 非合作关系设定下的多智能体强化学习 |                                                                                                                  |
| 17 注意力机制与多智能体强化学习       |                                                                                                                  |
| 18 AlphaGo 与蒙特卡洛树搜索           | [AlphaZero](https://github.com/suragnair/alpha-zero-general)                                                     |
| 19 现实世界中的应用                   | [NAS](https://github.com/titu1994/neural-architecture-search) [Recommender](https://github.com/awarebayes/RecNN) |


