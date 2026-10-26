import numpy as np
import os
import tensorflow as tf

from collections import deque
from tensorflow import keras
from keras import ops
from keras import layers

from threading import Lock

eps = np.finfo(np.float32).eps.item()

class ActorCriticAgent:
    NUM_PLAY_ACTIONS = 60
    NUM_BID_ACTIONS = 21

    def __init__(self, state_size: int,
                 checkpoint_path: str = "training_actor_critic/shared_policy.weights.h5"):
        self.state_size = state_size
        self.lock = Lock()
        self.lr = 3e-4
        self.gamma = 1.0
        self.max_grad_norm = 0.8
        self.ent_coef = 0.02
        self.phase_weight = {"play": 0.5, "bid": 0.5}
        self.epoch = 0
        self.memory_buffers = {}
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        state_in = keras.Input(shape=(self.state_size,), dtype=tf.float32, name="state")
        x = layers.Dense(256, activation="relu", name="shared_dense_1")(state_in)
        x = layers.Dense(256, activation="relu", name="shared_dense_2")(x)
        play_logits = layers.Dense(self.NUM_PLAY_ACTIONS,
                                   name="play_logits")(x)
        bid_logits = layers.Dense(self.NUM_BID_ACTIONS,
                                  name="bid_logits")(x)
        value_x = layers.Dense(128, activation="relu", name="value_dense_1")(x)
        value = layers.Dense(1, name="value")(value_x)
        self.model = keras.Model(
            state_in,
            [play_logits, bid_logits, value],
            name="wizard_shared_actor_critic",
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = keras.losses.Huber()

    def load_weights(self):
        if os.path.isfile(self.checkpoint_path):
            self.model.load_weights(self.checkpoint_path)
            print(f"Loaded shared play weights: {self.checkpoint_path}")

    def _legal_mask(self, allowed_actions, phase):
        size = self.NUM_BID_ACTIONS if phase == "bid" else self.NUM_PLAY_ACTIONS
        mask = np.zeros(size, dtype=bool)
        mask[allowed_actions] = True
        return mask

    def compute_action(self, current_state, allowed_actions, phase):
        mask = self._legal_mask(allowed_actions, phase)
        play_logits, bid_logits, value = self.model(
            ops.convert_to_tensor(current_state[None, :]), training=False)
        logits = play_logits if phase == "play" else bid_logits
        if np.isnan(np.sum(logits[0])):
            print(f"Something fishy!")
        masked_logits = tf.where(
            tf.convert_to_tensor(mask), logits[0],
            tf.cast(-1e9, logits.dtype))
        action = int(tf.random.categorical(masked_logits[None, :], 1)[0, 0])
        log_prob = float(tf.nn.log_softmax(masked_logits)[action])
        return action, log_prob, float(value[0, 0])

    def store_episode(self, current_state, action, legal, reward, next_state, done,
                      phase, actor_id, log_prob=None, value=None):
        with self.lock:
            if not actor_id in self.memory_buffers:
                self.memory_buffers[actor_id] = {"bid": deque(), "play": deque()}
            self.memory_buffers[actor_id][phase].append({
                "state": current_state, "action": action, "legal_actions": legal,
                "reward": reward, "next_state": next_state, "done": done,
                "log_prob": log_prob, "value": value
            })

    def evaluate_actions(self, states, masks, actions, phase):
        """Log-prob of the STORED actions under the current weights (call inside the tape)."""
        play_logits, bid_logits, values = self.model(states, training=True)
        logits = play_logits if phase == "play" else bid_logits
        masked = tf.where(masks, logits, tf.cast(-1e9, logits.dtype))
        log_probs_all = tf.nn.log_softmax(masked)
        log_prob = tf.gather(log_probs_all, actions, batch_dims=1)
        entropy = -tf.reduce_sum(tf.exp(log_probs_all) * log_probs_all, axis=-1)
        return log_prob, tf.squeeze(values, -1), entropy

    def _create_batch(self, rewards):
        # Create batch for a single model update pass
        batch = {phase: {"state": [], "legal": [], "action": [], "ret": [], "old_logp": []}
                     for phase in ("bid", "play")}
        for actor_history in self.memory_buffers.values():
            for phase, data in actor_history.items():
                while data and any(e["done"] for e in data):
                    # Events are stored in chronological order, so consume until "done"
                    events = []
                    while not (event := data.popleft())["done"]:
                        events.append(event)
                    rewards[phase].append(sum(e["reward"] for e in events))

                    # Calculate expected value from rewards
                    # - At each timestep what was the total reward received after that timestep
                    # - Rewards in the past are discounted by multiplying them with gamma
                    # - These are the labels for our critic
                    returns = []
                    discounted_sum = 0
                    for e in events[::-1]:
                        discounted_sum = e["reward"] + self.gamma * discounted_sum
                        returns.insert(0, discounted_sum)

                    b = batch[phase]
                    for e, ret in zip(events, returns):
                        b["state"].append(e["state"])
                        b["legal"].append(self._legal_mask(e["legal_actions"], phase))
                        b["action"].append(e["action"])
                        b["ret"].append(ret)
                        b["old_logp"].append(e.get("log_prob"))
        return batch

    def train(self):
        rewards = {"bid": [], "play": []}
        total_actor_losses = []
        total_critic_losses = []

        total_loss = None
        with tf.GradientTape() as tape:
            for phase, b in self._create_batch(rewards).items():
                states = tf.convert_to_tensor(np.array(b["state"]), tf.float32)
                masks = tf.convert_to_tensor(np.array(b["legal"], dtype=bool))
                actions = tf.convert_to_tensor(b["action"], tf.int32)
                ret = tf.convert_to_tensor(b["ret"], tf.float32)

                log_prob, value, entropy = self.evaluate_actions(states, masks, actions, phase)

                # Sanity check: weights are unchanged since acting, so these must match.
                if all(x is not None for x in b["old_logp"]):
                    max_diff = float(tf.reduce_max(tf.abs(log_prob - np.array(b["old_logp"],
                                                                              np.float32))))
                    if max_diff > 1e-4:
                        print(f"Warning: stored vs recomputed log_prob differ by {max_diff} ({phase})")

                # At this point in history, the critic estimated that we would get
                # a total reward = `value` in the future. We took an action with
                # log probability of `log_prob` and ended up receiving a total
                # reward = `ret`. The actor must be updated so that it predicts an
                # action that leads to high rewards (compared to critic's estimate)
                # with high probability.
                diff = tf.stop_gradient(ret - value)  # advantage must not train the critic
                diff = (diff - tf.reduce_mean(diff)) / \
                    (tf.math.reduce_std(diff) + eps)
                actor_loss = -tf.reduce_mean(log_prob * diff)

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_loss = tf.reduce_mean(self.loss(ret[:, None], value[:, None]))

                loss_value = self.phase_weight[phase] * (
                    actor_loss + 0.5 * critic_loss - self.ent_coef * tf.reduce_mean(entropy))

                total_actor_losses.append(float(actor_loss))
                total_critic_losses.append(float(critic_loss))
                total_loss = loss_value if total_loss is None else total_loss + loss_value

        # ---- 3. Single update for the whole game ----
        if total_loss is not None and np.isfinite(float(total_loss)):
            variables = self.model.trainable_variables
            grads = tape.gradient(total_loss, variables)
            pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
            clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], self.max_grad_norm)
            self.optimizer.apply_gradients(zip(clipped, [v for _, v in pairs]))
        else:
            print("Warning: skipped update (no data or non-finite loss)")

        mean = lambda x: np.mean(x) if len(x) else float("nan")
        print(f"Actor losses: {mean(total_actor_losses)}, "
              f"critic losses: {mean(total_critic_losses)}, "
              f"bid reward: {mean(rewards['bid'])} (std {np.std(rewards['bid'])}), "
              f"play reward: {mean(rewards['play'])} (std {np.std(rewards['play'])})")
        self.model.save_weights(self.checkpoint_path)
        self.epoch += 1
        self.memory_buffers.clear()
