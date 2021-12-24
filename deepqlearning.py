import numpy as np
from numpy.core.einsumfunc import _OptimizeKind
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)) 
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def remember(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min (self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dim, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
    input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
    mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(self.lr, self.n_actions, self.input_dims, 250, 250)
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, obs):
        """
        Choose action based on epsilon greedy policy and a prediction of the model
        Inferencer.
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([obs])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        """
        Q evaluation and learning, using the replay buffer.
        Inferencer.
        """
        if self.memory.mem_cntr < self.batch_size:
            return
        # Sample batch from memory
        states, actions, rewards, states_, terminal = self.memory.sample_buffer(self.batch_size)
        # Calculate q values
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Update Q-value for the selected action
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * terminal

        # Train the model
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
    
    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(self.model_file)

    