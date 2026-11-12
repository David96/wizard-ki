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

    def __init__(self, state_size: int, train: bool,
                 checkpoint_path: str = "training_actor_critic/shared_policy.weights.h5"):
        self.train = train
        self.state_size = state_size
        self.phase1 = False
        self.lock = Lock()
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_λ = 0.95
        self.clip_eps = 0.2
        self.ppo_epochs = 4
        self.n_minibatches = 4
        self.target_kl = 0.02
        self.max_grad_norm = 0.8
        self.ent_coef = 0.02
        self.phase_weight = {"play": 0.7, "bid": 0.3}
        self.epoch = 0
        self.memory_buffers = {}
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        state_in = keras.Input(shape=(self.state_size,), dtype=tf.float32, name="state")
        x = layers.Dense(512, activation="relu", name="shared_dense_1")(state_in)
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
        if self.train:
            action = int(tf.random.categorical(masked_logits[None, :], 1)[0, 0])
        else:
            action = int(tf.math.margmax(masked_logits))
        log_prob = float(tf.nn.log_softmax(masked_logits)[action])
        return action, log_prob, float(value[0, 0])

    def store_episode(self, current_state, action, legal, reward, next_state, done,
                      phase, actor_id, log_prob, value):
        if not self.train:
            return
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

    def compute_gae(self, events):
        """One episode in time order. Returns (advantages, critic targets), both on the scaled reward."""
        values = np.array([e["value"] for e in events], np.float32)
        rewards = np.array([e["reward"] for e in events], np.float32)
        next_values = np.append(values[1:], 0.0)  # value after the terminal step is 0
        deltas = rewards + self.gamma * next_values - values
        adv = np.zeros_like(deltas)
        last = 0.0
        for t in reversed(range(len(deltas))):
            last = deltas[t] + self.gamma * self.gae_λ * last
            adv[t] = last
        return adv, adv + values   # critic target = advantage + old value

    def _create_batch(self, rewards):
        # Create batch for a single model update pass
        tmp_batch = {p: {"state": [], "legal": [], "action": [], "adv": [],
                         "rewards": [], "ret": [], "old_logp": []} for p in ["bid", "play"]}
        batch = {"bid": {}, "play": {}}
        for actor_history in self.memory_buffers.values():
            for phase, data in actor_history.items():
                b = tmp_batch[phase]
                while data and any(e["done"] for e in data):
                    # Events are stored in chronological order, so consume until "done"
                    events = []
                    while not (event := data.popleft())["done"]:
                        events.append(event)
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

                    adv, ret = self.compute_gae(events)
                    for e, a, r in zip(events, adv, ret):
                        b["state"].append(e["state"])
                        b["legal"].append(self._legal_mask(e["legal_actions"], phase))
                        b["action"].append(e["action"])
                        b["adv"].append(a)
                        b["ret"].append(r)
                        b["rewards"].append(e["reward"])
                        b["old_logp"].append(e["log_prob"])
        for p, b in tmp_batch.items():
            adv = np.array(b["adv"])
            # Standardize advantages across the pooled batch, but only scale
            # when the spread is meaningful. Near-constant batches (e.g. a
            # phase with no trick rewards) would otherwise be normalized into
            # pure noise.
            if adv.size and adv.std() > 1e-3:
                adv = (adv - adv.mean()) / (adv.std() + eps)
            elif adv.size:
                adv = adv - adv.mean()
            batch[p]["adv"] = tf.convert_to_tensor(adv, tf.float32)
            batch[p]["rewards"] = tf.convert_to_tensor(b["rewards"], tf.float32)
            batch[p]["legal"] = tf.convert_to_tensor(np.array(b["legal"], dtype=bool))
            batch[p]["action"] = tf.convert_to_tensor(b["action"], tf.int32)
            batch[p]["ret"] = tf.convert_to_tensor(b["ret"], tf.float32)
            batch[p]["state"] = tf.convert_to_tensor(np.array(b["state"]), tf.float32)
            batch[p]["old_logp"] = tf.convert_to_tensor(b["old_logp"], tf.float32)
        return batch

    def train(self):
        if not self.train:
            return
        rewards = {"bid": [], "play": []}
        total_player_losses = []
        total_bids_losses = []
        total_critic_losses = []
        approx_kls, clip_fracs = [], []
        batch = self._create_batch(rewards)
        for epoch in range(self.ppo_epochs):
            splits = {p: np.array_split(np.random.permutation(len(x["action"])),
                                        self.n_minibatches)
                          for p, x in batch.items()}
            epoch_kl = []
            for k in range(self.n_minibatches):
                total_loss = None
                with tf.GradientTape() as tape:
                    for phase, b in batch.items():
                        idx = splits[phase][k]
                        state = tf.gather(b["state"], idx)
                        reward = tf.gather(b["rewards"], idx)
                        legal = tf.gather(b["legal"], idx)
                        action = tf.gather(b["action"], idx)
                        old_logp = tf.gather(b["old_logp"], idx)
                        adv = tf.gather(b["adv"], idx)
                        ret = tf.gather(b["ret"], idx)
                        # Train bid head from the cross entropy to actual bids won
                        if phase == "bid" and self.phase1:
                            _, bid_logits, _ = self.model(state, training=True)
                            ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                                reward, bid_logits, from_logits=True))
                            total_bids_losses.append(ce)
                            loss_value = self.phase_weight[phase] * ce
                            total_loss = loss_value if total_loss is None else total_loss + loss_value
                            continue

                        log_prob, value, entropy = self.evaluate_actions(state, legal,
                                                                         action, phase)

                        if epoch == 0 and k == 0:
                            # Sanity check: weights are unchanged since acting,
                            # so these must match.
                            max_diff = float(tf.reduce_max(tf.abs(log_prob - old_logp)))
                            if max_diff > 1e-4:
                                print(f"Warning: stored vs recomputed log_prob differ by {max_diff} ({phase})")

                        # PPO clipped objective
                        ratio = tf.exp(log_prob - old_logp)
                        clipped = tf.clip_by_value(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        actor_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped *
                                                                adv))
                        critic_loss = tf.reduce_mean(self.loss(ret[:, None], value[:, None]))
                        loss_value = self.phase_weight[phase] * (
                            actor_loss + 0.5 * critic_loss - self.ent_coef * tf.reduce_mean(entropy))

                        total_player_losses.append(float(actor_loss))
                        total_critic_losses.append(float(critic_loss))
                        total_loss = loss_value if total_loss is None else total_loss + loss_value
                        epoch_kl.append(float(tf.reduce_mean(old_logp - log_prob)))
                        clip_fracs.append(float(tf.reduce_mean(
                            tf.cast(tf.abs(ratio - 1) > self.clip_eps, tf.float32))))

                # ---- 3. Single update for the whole game ----
                if total_loss is not None and np.isfinite(float(total_loss)):
                    variables = self.model.trainable_variables
                    grads = tape.gradient(total_loss, variables)
                    pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
                    clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], self.max_grad_norm)
                    self.optimizer.apply_gradients(zip(clipped, [v for _, v in pairs]))
                else:
                    print("Warning: skipped update (no data or non-finite loss)")
            approx_kls.append(np.mean(epoch_kl) if epoch_kl else 0.0)
            if approx_kls[-1] > 1.5 * self.target_kl:   # policy already moved enough
                break

        mean = lambda x: np.mean(x) if len(x) else float("nan")
        print(f"Player losses: {mean(total_player_losses)}, "
              f"Bids losses: {mean(total_bids_losses)}, "
              f"critic losses: {mean(total_critic_losses)}, "
              f"approx KL: {mean(approx_kls)}, clip frac: {mean(clip_fracs)}, "
              f"bid reward: {mean(rewards['bid'])} (std {np.std(rewards['bid'])}), "
              f"play reward: {mean(rewards['play'])} (std {np.std(rewards['play'])})")
        with open("training_data.txt", "ab") as f:
            np.savetxt(f, [mean(total_player_losses), mean(total_bids_losses),
                           mean(total_critic_losses)], fmt="%.5f", delimiter=' ', newline=', ')
            f.write(b'\n')
        self.model.save_weights(self.checkpoint_path)
        self.epoch += 1
        self.memory_buffers.clear()
