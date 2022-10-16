"""14.3节MPE环境。
安装依赖环境：pip install "pettingzoo[mpe]"
"""

from pettingzoo.mpe import simple_crypto_v2
import time

env = simple_crypto_v2.env()

num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).n
observation_size = env.observation_space(env.possible_agents[0]).shape

print(f"{num_agents} agents")
for i in range(num_agents):
    num_actions = env.action_space(env.possible_agents[i]).n
    observation_size = env.observation_space(env.possible_agents[i]).shape
    print(i, env.possible_agents[i], "num_actions:", num_actions, "observation_size:", observation_size)


env.reset()
for i, agent in enumerate(env.agent_iter()):
    observation, reward, termination, info = env.last()
    action = 0

    action = env.action_space(agent).sample()
    env.step(action)

    print(i, agent)
    print(f"action={action}, observation={observation}, reward={reward}, termination={termination}, info={info}")

    env.render()
    time.sleep(0.1)

    if i == 50:
        break
