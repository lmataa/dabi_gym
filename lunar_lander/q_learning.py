import numpy as np
from IPython.display import clear_output
from tqdm import tqdm
import sys

def q_table_discrete(state_space, action_space, bin_size = 30):
    """
    Creates a Q-table for the given state and action spaces.
    """
    bins = []
    bins.append(np.linspace(-3, -0.1, bin_size))
    bins.append(np.linspace(-0.1, -0.05, bin_size))
    bins.append(np.linspace(-0.05, -0.025, bin_size))
    bins.append(np.linspace(-0.025, -0.0125, bin_size))
    bins.append(np.linspace(-0.0125, 0, bin_size))
    bins.append(np.linspace(0, 0.2, bin_size))
    bins.append(np.linspace(0.2, 0.55, bin_size))
    bins.append(np.linspace(0.55, 3, bin_size))

    q_table = np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))
    return q_table, bins

def discrete(state, bins):
    index = []
    for i in range(len(state)): 
        index.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(index)


def q_learning(env, alpha=0.7, gamma=0.5, epsilon=0.5, epsilon_decay=0.0019, epsilon_min=0.01, episodes=500, bin_size = 9, render=False):
    # q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])
    print("Allocating memory for Q-table ...")
    q_table, bins = q_table_discrete(env.observation_space.shape[0], env.action_space.n, bin_size=bin_size)
    print(q_table.shape)
    print(len(bins))
    print(len(env.reset()))
    scores = []
    for episode in tqdm(range(episodes), file=sys.stdout):
        #state = env.reset()
        state = discrete(env.reset(), bins) 
        #print(state)
        reward = 0
        score = 0
        terminated = False
        while not terminated:
        # Take learned path or explore new actions based on the epsilon
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action    
            observation, reward, terminated, info = env.step(action)
            next_state = discrete(observation, bins)
            #print(observation)
            #print(next_state)
            score += reward
            # Recalculate
            q_value = q_table[state, action]
            max_value = np.max(q_table[next_state])
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)
            
            # Update Q-table
            q_table[state, action] = new_q_value
            state = next_state
            epsilon = epsilon - epsilon_decay if epsilon > epsilon_min else epsilon_min
        scores.append(score)
        
    if (episode + 1) % 100 == 0:
        clear_output(wait=True)
        avg_score = np.mean(scores[-100:])
        print(f"Episode: {episode + 1}")
        print(f"Info: {info}")
        print(f"Score: {score}")
        print(f"Reward: {reward}")
        print(f"Action taken: {action}")
        print(f"Average score: {avg_score}")
        env.render()
    print("**********************************")
    print("Training is done!\n")
    print("**********************************")
    return q_table, scores

def test_q_learning(env, q_table, episodes=10, patience=100, bin_size=6, render=False):
    total_penalties = 0
    total_epochs = 0
    scores = []
    _, bins = q_table_discrete(env.observation_space.shape[0], env.action_space.n, bin_size=bin_size)
    for _ in range(episodes):
        state = discrete(env.reset(), bins)
        epochs = 0
        penalties = 0
        reward = 0
        score = 0
        terminated = False
        
        while not terminated:
            action = np.argmax(q_table[state])
            observation, reward, terminated, info = env.step(action)
            state = discrete(observation, bins)
            print(reward)
            print(info)
            score += reward
            if reward <= 0:
                penalties += 1
            
            epochs += 1
            if (epochs + 1) % 1 == 0:
                clear_output(wait=True)
                print("Episode: {}".format(epochs+ 1))
                env.render()

            if epochs > patience:
                terminated = True
        total_penalties += penalties
        total_epochs += epochs
        scores.append(score)
    avg_score = np.mean(scores[-100:])
    print("**********************************")
    print("Results")
    print("**********************************")
    print("Epochs per episode: {}".format(total_epochs / episodes))
    print("Penalties per episode: {}".format(total_penalties / episodes))
    return total_penalties / episodes, avg_score