import numpy as np
import gym
from q_learning import q_learning, test_q_learning
from utils import plot_learning_curve, QLearningParser
from pathlib import Path
import pickle

def main(args=None):
    env = gym.make('LunarLander-v2')
    output_path = args.path/args.exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    if not args.test:
        q_table, scores = q_learning(env,
                                alpha=args.alpha, 
                                gamma=args.gamma, 
                                epsilon=args.epsilon, 
                                epsilon_decay=args.decay,
                                epsilon_min=args.epsilon_min,
                                episodes=args.epochs, 
                                bin_size=args.bins,
                                render=False)

        penalty_ratio, avg_score = test_q_learning(env, q_table, 
                                                    episodes=10, 
                                                    patience=args.patience, 
                                                    render=False)

        with open(output_path/"metrics.pkl", "wb") as f:
            pickle.dump({"penalty_ratio": penalty_ratio, "avg_score": avg_score}, f)

        np.save(output_path/"q_table.npy", q_table)
        # Plot learning curve
        file = output_path/"training_qlearning.png"
        x = [i+1 for i in range(args.epochs)]
        plot_learning_curve(x, scores, None, file)
    else:
        q_table = np.load(output_path/"q_table.npy")
        penalty_ratio, avg_score = test_q_learning(env, q_table, 
                                                    episodes=10, 
                                                    patience=args.patience, 
                                                    render=False)


if __name__ == '__main__':
    EXP_NAME = "test_q_learning_" + np.datetime64('now').astype(str)
    DEFAULT_OUTPUT_PATH = Path(__file__).parent/ 'output'
    DEFAULT_EPOCHS = 1000
    DEFAULT_ALPHA = 0.7
    DEFAULT_GAMMA = 0.5
    DEFAULT_EPSILON = 1
    DEFAULT_EPSILON_DECAY = 0.0001
    DEFAULT_EPSILON_MIN = 0.01
    DEFAULT_PATIENCE = 100
    DEFAULT_BINS = 10

    parser = QLearningParser(
        defaults={
            'path': DEFAULT_OUTPUT_PATH,
            'exp_name': EXP_NAME,
            'epochs': DEFAULT_EPOCHS,
            'alpha': DEFAULT_ALPHA,
            'gamma': DEFAULT_GAMMA,
            'epsilon': DEFAULT_EPSILON,
            'epsilon_decay': DEFAULT_EPSILON_DECAY,
            'epsilon_min': DEFAULT_EPSILON_MIN,
            'patience': DEFAULT_PATIENCE,
            'bins': DEFAULT_BINS
        })
    args = parser.parse_args()
    parser.handle(args)
    main(args)