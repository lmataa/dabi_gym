from dqn import Agent
import numpy as np
import gym
import tensorflow as tf
from utils import plot_learning_curve

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    n_actions = env.action_space.n
    input_dims = env.observation_space.shape
    lr = 0.001

    print(input_dims)
    agent = Agent(gamma=0.99, epsilon=1.0, epsilon_dec=0.00001955, epsilon_end=0.01,
                    lr=lr, batch_size=64, n_actions=n_actions,
                    input_dims=input_dims, mem_size=1000000,
                    fname='lunar_lander.h5')
    scores = []
    eps_history = []
    num_games = 2500
    for i in range(num_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            # Temporal diference learn
            agent.learn() 
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon % .2f' % agent.epsilon)
    agent.save_model()
    filename = "lunalander_tf2.png"
    x=[i+1 for i in range(num_games)]
    plot_learning_curve(x, scores, eps_history, filename)
