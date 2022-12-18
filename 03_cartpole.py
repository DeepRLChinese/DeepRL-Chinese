"""3.6节实验环境。
"""


import gym


env = gym.make("CartPole-v0", render_mode="human")
state = env.reset()


for t in range(1000):
    env.render()
    print(state)

    action = env.action_space.sample()

    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Finished")
        state = env.reset()

env.close()
