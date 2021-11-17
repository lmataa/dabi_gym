import argparse
import gym
import numpy as np

def main(raw_args=None):
    # env = gym.make('FrozenLake8x8-v1')
    # env = gym.make('CartPole-v0')
    # env = gym.make('Go9x9-v0')
    # env = gym.make('MsPacman-v0')
    # env = gym.make('Zaxxon-v4')
    # env = gym.make('Assault-v0')
    # env = gym.make('Breakout-ram-v0')
    # env = gym.make('Freeway-v0')
    env = gym.make('SpaceInvaders-v0')
    print(env.action_space)
    print(env.observation_space.shape)
    print()
    for e in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = np.random.choice([4, 5]) #env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print(f"Episode {e} finished after {t+1} timesteps")
                break
            if raw_args.debug and t%5==0:
                input("Press Enter to continue...")
            #env.step(env.action_space.sample()) # take a random action
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            default=False,
            help="Debug mode.")
    args = parser.parse_args()
    main(args)
