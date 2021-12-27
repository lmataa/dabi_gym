"""
Script to test LunarLander environment with DQN agent.
"""
from numpy.core.fromnumeric import choose
from dqn import Agent
import numpy as np
import gym
import tensorflow as tf
import argparse
from pathlib import Path

def choose_action(eps, obs, model, action_space):
        """
        Choose action based on epsilon greedy policy and a prediction of the model
        Inferencer.
        """
        if np.random.random() < eps:
            action = np.random.choice(action_space)
        else:
            state = np.array([obs])
            actions = model.predict(state)
            action = np.argmax(actions)
        return action

def make_video(env, model, epsilon_greedy, path, random):
    """
    Make a video of the game, using the model or with random actions to compare.
    """
    env = gym.wrappers.Monitor(env, path, force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        if random:
            action = np.random.choice(range(env.action_space.n))
        elif epsilon_greedy:
            epsilon = 0.1
            action_space = [i for i in range(env.action_space.n)]
            action = choose_action(epsilon, observation, model, action_space)
        else:
            action = np.argmax(model.predict(np.array([observation])))
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))        

def render_game(env, epsilon_greedy, q_eval, debug=False):
    """
    Play 20 games and show the results while rendering them.
    """
    for e in range(20):
        observation = env.reset()
        score = 0
        scores = []
        for t in range(1000):
            env.render()
            if epsilon_greedy:
                epsilon = 0.1
                action_space = [i for i in range(env.action_space.n)]
                action = choose_action(epsilon, observation, q_eval, action_space)
            else:
                action = np.argmax(q_eval.predict(np.array([observation])))
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
            if debug and t%5==0:
                input("Press Enter to continue...")
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', e, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'finished after: %d steps' % t)
        env.close()

def main(raw_args=None):
    env = gym.make('LunarLander-v2')
    model_path = args.path
    q_eval = tf.keras.models.load_model(model_path)
    epsilon_greedy = args.epsilon_greedy
    if not args.make_video:
        render_game(env, epsilon_greedy, q_eval, args.debug)
    else:
        video_path = Path(args.path).parent/"videos"
        video_path.mkdir(exist_ok=True)
        make_video(env, q_eval, epsilon_greedy, video_path, args.random)

if __name__ == "__main__":
    exp_name = 'test_02'
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            default=False,
            help="Debug mode.")
    parser.add_argument(
            "-m",
            "--make_video",
            action="store_true",
            default=False,
            help="Show one and just make video.")
    parser.add_argument(
            "-r",
            "--random",
            action="store_true",
            default=False,
            help="Random action.")
    parser.add_argument(
            "-e",
            "--epsilon_greedy",
            action="store_true",
            default=False,
            help="Epsilon greedy action selection.")
    parser.add_argument(
            "-path",
            nargs="?",
            default=Path("output/")/exp_name/"lunar_lander.h5",
            help="Path to create the file system.")
    args = parser.parse_args()
    main(args)