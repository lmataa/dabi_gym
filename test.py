import gym
# env = gym.make('FrozenLake8x8-v1')
# env = gym.make('CartPole-v0')
env = gym.make('Go9x9-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

