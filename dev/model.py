
import tensorflow as tf 
import numpy as np 
from pathlib import Path

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.95 # decay rate of past observations
OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
REPLAY_MEMORY = 40000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 10000

OUTPUT_DIR = Path("./output/")


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.extra_layers = [
             tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='RandomNormal'),
             tf.keras.layers.Dense(num_actions, activation='tanh', kernel_initializer='RandomNormal'),
        ]
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        print("input_shape", z.shape)
        for layer in self.hidden_layers:
            z = layer(z)
        #for layer in self.extra_layers:
        #    z = layer(z)
        output = self.output_layer(z)
        print(output.shape)
        return output

class CNN_Model(tf.keras.Model):
    def __init__(self, input_shape, num_states, hidden_units, num_actions):
        super(CNN_Model, self).__init__()
        self.seq_model = tf.keras.Sequential(
            [ 
            #tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(input_shape=input_shape, filters=32, activation='relu', kernel_size=8, strides=4, data_format="channels_last"),
            tf.keras.layers.Conv2D(filters=64, activation='relu', kernel_size=4, strides=2, padding='same'),
            tf.keras.layers.Conv2D(filters=64, activation='relu', kernel_size=3, strides=1),
            tf.keras.layers.Dense(512, activation='linear'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal'),
            ]
        )
        self.seq_model.build()
        self.seq_model.summary()
        
    @tf.function
    def call(self, inputs):
        return self.seq_model(inputs)
    
class DQN:
    
    def __init__(self, exp_name, output_dir, model, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        
        # Object handling
        self.exp_name=exp_name
        self.mkdirs(output_dir)
        
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = model #MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))
    
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return loss
    
    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])
        
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
    
    
    def mkdirs(self, output_dir):
        # Output Paths: metrics, models, ds_dir
        self.output_dir = output_dir / self.exp_name
        self.metrics_dir = self.output_dir / "metrics"
        self.models_dir = self.output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self):
        self.model.save(self.models_dir/f"{self.exp_name}.h5")
        pass