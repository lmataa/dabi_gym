from dqn import Agent
import numpy as np
import gym
import tensorflow as tf
from utils import plot_learning_curve, DefaultParser
from pathlib import Path
import pickle

def main(args=None):
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    n_actions = env.action_space.n
    input_dims = env.observation_space.shape
    output_path = args.path/args.exp_name
    output_path.mkdir(parents=True, exist_ok=True)

    agent = Agent(gamma=args.gamma,
                    epsilon=args.epsilon,
                    epsilon_dec=args.decay,
                    epsilon_end=args.epsilon_min,
                    lr=args.learning_rate, 
                    batch_size=args.batch_size,
                    n_actions=n_actions,
                    input_dims=input_dims,
                    mem_size=args.memory_size,
                    output_path=output_path,
                    fname='lunar_lander.h5')
    scores = []
    eps_history = []
    x=[i+1 for i in range(args.epochs)]
    metrics_dict = {}
    for i in range(args.epochs):
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
        if i % 200 == 0:
            agent.save_model(i)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon % .2f' % agent.epsilon)
    # Save model
    agent.save_model(args.epochs)
    # Save metrics
    metrics_dict["scores"] = scores
    metrics_dict["eps_history"] = eps_history
    metrics_dict["x"] = x
    with open(output_path/'metrics.pkl', 'wb') as f:
        pickle.dump(metrics_dict, f)
    # Plot learning curve
    file = output_path/"lunalander_tf2.png"
    plot_learning_curve(x, scores, eps_history, file)


if __name__ == '__main__':
    EXP_NAME = "DQN_LunarLander_000"
    DEFAULT_OUTPUT_PATH = Path(__file__).parent/ 'output'
    DEFAULT_EPOCHS = 10000
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_GAMMA = 0.99
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPSILON = 1.0
    DEFAULT_EPSILON_DECAY = 0.00001955
    DEFAULT_EPSILON_MIN = 0.01
    DEFAULT_MEMORY_SIZE = 1000000

    parser = DefaultParser(
        defaults={
            'path': DEFAULT_OUTPUT_PATH,
            'exp_name': EXP_NAME,
            'epochs': DEFAULT_EPOCHS,
            'batch_size': DEFAULT_BATCH_SIZE,
            'gamma': DEFAULT_GAMMA,
            'learning_rate': DEFAULT_LEARNING_RATE,
            'epsilon': DEFAULT_EPSILON,
            'epsilon_decay': DEFAULT_EPSILON_DECAY,
            'epsilon_min': DEFAULT_EPSILON_MIN,
            'memory_size': DEFAULT_MEMORY_SIZE
        })
    args = parser.parse_args()
    parser.handle(args)
    main(args)