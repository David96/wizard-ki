import tensorflow as tf
import numpy as np
import os
import os.path

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

checkpoint_dir = "training_2/"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#        save_weights_only=True,
#        verbose=1)

class QAgent:

    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 0.6
        self.exploration_proba_decay = 0.05
        self.batch_size = 32
        self.epoch = 0

        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        self.model = Sequential([
                Dense(units=24, input_dim=state_size, activation="relu"),
                Dense(units=24, activation="relu"),
                Dense(units=action_size, activation="linear")
            ])
        self.model.compile(loss="mse", optimizer=Adam(lr=self.lr))

    def load_weights(self):
        if os.path.isfile(checkpoint_dir + str(self.n_actions) + ".tf.index"):
            self.model.load_weights(checkpoint_dir + str(self.n_actions) + ".tf")
            print("Loaded weights")

    def compute_action(self, current_state, allowed_actions, force_random=False):
        if force_random or np.random.uniform(0,1) < self.exploration_proba:
            return int(np.random.choice(allowed_actions))
        q_values = self.model.predict(np.array([current_state]))[0]
        sorted_q = (-q_values).argsort()
        print("q_values: %s" % q_values)
        print("Sorted actions: %s" % sorted_q)
        for action in sorted_q:
            if int(action) in allowed_actions:
                print("Found valid action: %d" % int(action))
                return int(action)
        return 0

    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
            })
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    def train(self):
        np.random.shuffle(self.memory_buffer)

        for exp in self.memory_buffer:
            current_state = exp["current_state"].reshape(1, -1)
            q_current_state = self.model.predict(current_state)
            q_target = exp["reward"]
            if not exp["done"]:
                next_state = exp["next_state"].reshape(1, -1)
                q_target = q_target + self.gamma *\
                    np.max(self.model.predict(next_state)[0])
            q_current_state[0][exp["action"]] = q_target
            self.model.fit(current_state, q_current_state, initial_epoch=self.epoch,
                    verbose=1, epochs=self.epoch + 1)

        self.model.save_weights(checkpoint_dir + str(self.n_actions) + ".tf")
        self.epoch += 1
