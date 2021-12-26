"""
Script to test LunarLander environment with DQN agent.
"""
from dqn import Agent
import numpy as np
import gym
import tensorflow as tf
import argparse
from pathlib import Path

def main(raw_args=None):
    env = gym.make('LunarLander-v2')
    exp_name = 'test_02'
    model_path = Path("output/")/exp_name/"lunar_lander.h5"
    q_eval = tf.keras.models.load_model(model_path)

    for e in range(20):
        observation = env.reset()
        score = 0
        scores = []
        for t in range(1000):
            env.render()
            action = np.argmax(q_eval.predict(np.array([observation])))
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