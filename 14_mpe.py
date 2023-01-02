"""14.3节MPE环境。
安装依赖环境：pip install "pettingzoo[mpe]"
"""

from pettingzoo.mpe import simple_spread_v2
import time

env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode="human")

num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).n
observation_size = env.observation_space(env.possible_agents[0]).shape

print(f"{num_agents} agents")
for i in range(num_agents):
    num_actions = env.action_space(env.possible_agents[i]).n
    observation_size = env.observation_space(env.possible_agents[i]).shape
    print(i, env.possible_agents[i], "num_actions:", num_actions, "observation_size:", observation_size)

for epoch in range(3):
    env.reset()
    for i, agent in enumerate(env.agent_iter()):
        observation, reward, terminated, truncated, info = env.last()
        done = terminated or truncated
        action = 0

        if done:
            break

        action = env.action_space(agent).sample()
        env.step(action)

        print(i, agent)
        print(f"action={action}, observation={observation}, reward={reward}, done={done}, info={info}")

        time.sleep(3)
