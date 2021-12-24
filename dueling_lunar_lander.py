from dueling_dqn import Agent
import numpy as np
import gym
from lunar_lander import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    scores = []
    eps_history = []
    num_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, epsilon_dec=1e-3, epsilon_end=0.01,
                    lr=0.001, batch_size=64, n_actions=env.action_space.n,
                    input_dims=env.observation_space.shape, mem_size=1000000)
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
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %1f' % score,
                'average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        filename = 'lunar_lander_dueling.png'
        x = [i+1 for i in range(len(num_games))]
        plot_learning_curve(x, scores, eps_history, filename)
        