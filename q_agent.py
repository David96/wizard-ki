import tensorflow as tf
import numpy as np
import os
import os.path

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

checkpoint_dir = "training_9/"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#        save_weights_only=True,
#        verbose=1)

class QAgent:

    def __init__(self, state_size, action_output_size, bidding_output_size env):
        self.env = env
        self.n_output_action= action_output_size
        self.n_output_bidding= bidding_output_size
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 0.1
        self.exploration_proba_decay = 0.01
        self.batch_size = 32
        self.epoch = 0

        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        self.encoder= Sequential([
                Input(state_size),
                Dense(512, activation="relu"),
                Dense(512, activation="relu"),
            ])
        self.action_head = Dense(action_output_size)
        self.bidding_head = Dense(bidding_output_size)
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))

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
        return self.get_first_allowed_action(sorted_q, allowed_actions)

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

    def get_first_allowed_action(self, sorted_q, allowed_actions):
        for action in sorted_q:
            if int(action) in allowed_actions:
                return int(action)

    def train(self):
        self.load_weights()
        np.random.shuffle(self.memory_buffer)
        #sample = self.memory_buffer[0:min(len(self.memory_buffer) - 1, 64)]

        for exp in self.memory_buffer:
            current_state = exp["current_state"].reshape(1, -1)
            q_current_state = self.model.predict(current_state)
            q_target = exp["reward"]
            if not exp["done"]:
                next_state = exp["next_state"].reshape(1, -1)
                state_prediction = (-self.model.predict(next_state)[0]).argsort()
                action_prediction = self.get_first_allowed_action(state_prediction,
                        self.env.get_valid_actions(next_state[0]))
                q_target = q_target + self.gamma * state_prediction[action_prediction]

            q_current_state[0][exp["action"]] = q_target
            self.model.fit(current_state, q_current_state, initial_epoch=self.epoch,
                    verbose=1, epochs=self.epoch + 1)

        self.model.save_weights(checkpoint_dir + str(self.n_actions) + ".tf")
        self.epoch += 1
