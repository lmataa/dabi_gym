import tensorflow as tf
import tensorlfow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        # value head
        self.V = keras.layer.Dense(1, activation=None)
        # advantage layer
        self.A = keras.layer.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        # value head
        V = self.V(x)
        # advantage head
        A = self.A(x)
        # non trivial Q value
        Q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        """
        Helper for choosing action, 
        """
        x = self.dense1(state)
        x = self.dense2(x)
        # advantage head
        A = self.A(x)
        return A

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
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

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
    input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
    mem_size=100000, fname='dueling_dqn.h5', fc1_dims=128, fc2_dims=128, replace=100):
        """
        LR: learning rate
        Gamma: discount factor
        Epsilon: probability of random action
        Epsilon_dec: decay rate of epsilon
        Epsilon_end: final value of epsilon
        Replace: how many steps to replace target network, helps to stabilize learning
        """
        self.action_space = [i for i in range(n_actions)]
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace=replace

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions)
        self.q_next = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions)

        self.q_eval.compile(optimizer=Adam(lr=self.lr), loss='mse')
        # just a formality, won't optimize next network
        self.q_next.compile(optimizer=Adam(lr=self.lr), loss='mse')

    def remember(self, state, action, reward, new_state, done): 
        self.memory.remember(state, action, reward, new_state, done)
    
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = state[np.newaxis, :]
            actions = self.q_eval.advantage(state)
            #action = tf.math.argmax(actions, axis=1).numpy()[0]
            action = np.argmax(actions)
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
        states, action, reward, new_state, dones = self.memory.sample_buffer(self.batch_size)
        
        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(new_state), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        # improve
        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx, action[idx]] = reward[idx] + self.gamma * q_next[idx]

        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(self.model_file)
