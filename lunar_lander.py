from deepqlearning import Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    running_avg = [np.mean(scores[max(0, t-20):(t+1)]) for t in range(N)]

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y'. colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
    plt.savefig(filename)

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    n_actions = env.action_space.n
    input_dims = env.observation_space.shape
    lr = 0.001

    print(input_dims)
    agent = Agent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, epsilon_end=0.01,
                    lr=lr, batch_size=64, n_actions=n_actions,
                    input_dims=input_dims, mem_size=1000000,
                    fname='lunar_lander.h5')
    scores = []
    eps_history = []
    num_games = 500
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
    filename = "lunalander_tf2.png"
    x=[i+1 for i in range(len(num_games))]
    plot_learning_curve(x, scores, eps_history, filename)