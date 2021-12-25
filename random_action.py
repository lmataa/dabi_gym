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
    # env = gym.make('SpaceInvaders-v4')
    env = gym.make('LunarLander-v2')
    # env = gym.make('Pong-v4')
    print(env.action_space)
    print(env.observation_space.shape)
    scores = []
    for e in range(20):
        observation = env.reset()
        score = 0
        for t in range(1000):
            env.render()
            action = np.random.choice(range(env.action_space.n)) #env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
            if raw_args.debug and t%5==0:
                input("Press Enter to continue...")
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', e, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'finished after: %d steps' % t)
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
