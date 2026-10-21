"""Shared masked PPO actor-critic for Wizard.

One instance is shared by every WizardEnv/player. The trunk and value head are
shared; bidding and playing have separate categorical policy heads (21 and 60
actions). Rollouts are collected with frozen weights and optimized only after
all players' games finish.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class QAgent:
    """Compatibility name retained; this is a shared PPO policy, not a Q-agent."""

    NUM_PLAY_ACTIONS = 60
    NUM_BID_ACTIONS = 21

    def __init__(self, state_size: int, action_size: Optional[int] = None,
                 env=None, checkpoint_path: str = "training_ppo/shared_policy.weights.h5"):
        # action_size/env remain optional-compatible with the old constructor.
        self.state_size = int(state_size)
        self.lr = 1e-3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.02
        self.batch_size = 128
        self.ppo_epochs = 4
        self.max_grad_norm = 0.7
        self.max_memory_buffer = 200_000
        self.epoch = 0
        self.memory_buffer = []
        self._pending = {}  # (actor_id, phase) -> action metadata
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        state_in = keras.Input(shape=(self.state_size,), dtype=tf.float32, name="state")
        x = layers.Dense(256, activation="relu", name="shared_dense_1")(state_in)
        x = layers.Dense(256, activation="relu", name="shared_dense_2")(x)
        x = layers.Dense(128, activation="relu", name="shared_dense_3")(x)
        play_logits = layers.Dense(self.NUM_PLAY_ACTIONS, name="play_logits")(x)
        bid_logits = layers.Dense(self.NUM_BID_ACTIONS, name="bid_logits")(x)
        value = layers.Dense(1, name="value")(x)
        self.model = keras.Model(
            state_in,
            {"play_logits": play_logits, "bid_logits": bid_logits, "value": value},
            name="wizard_shared_actor_critic",
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

    @staticmethod
    def _as_batch(state):
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 1-D state, got shape {arr.shape}")
        return arr

    @staticmethod
    def _phase_name(phase):
        if phase in ("announce", "announcing", "bid", "bidding"):
            return "bid"
        if phase in ("play", "playing", "play_card"):
            return "play"
        raise ValueError(f"Unknown Wizard phase: {phase!r}")

    def _forward(self, states, training=False):
        return self.model(tf.convert_to_tensor(states, dtype=tf.float32), training=training)

    def _legal_mask(self, allowed_actions, phase):
        size = self.NUM_BID_ACTIONS if phase == "bid" else self.NUM_PLAY_ACTIONS
        actions = np.asarray(list(allowed_actions), dtype=np.int64)
        if actions.size == 0:
            raise ValueError(f"No legal {phase} actions supplied")
        if np.any(actions < 0) or np.any(actions >= size):
            raise ValueError(f"Illegal action index for {phase} head (size {size}): {actions}")
        mask = np.zeros(size, dtype=bool)
        mask[actions] = True
        return mask

    def load_weights(self):
        if os.path.isfile(self.checkpoint_path):
            self.model.load_weights(self.checkpoint_path)
            print(f"Loaded shared PPO weights: {self.checkpoint_path}")

    def compute_action(self, current_state, allowed_actions,
                       phase="play", actor_id="default"):
        phase = self._phase_name(phase)
        key = (str(actor_id), phase)
        mask = self._legal_mask(allowed_actions, phase)

        # A new decision supersedes any unconsumed metadata for this actor/phase.
        self._pending.pop(key, None)

        output = self._forward(self._as_batch(current_state), training=False)
        logits = output["bid_logits" if phase == "bid" else "play_logits"][0]
        value = tf.squeeze(output["value"], axis=-1)[0]
        masked_logits = tf.where(
            tf.convert_to_tensor(mask), logits,
            tf.cast(-1e9, logits.dtype))
        log_probs = tf.nn.log_softmax(masked_logits)
        action = int(tf.random.categorical(log_probs[None, :], 1)[0, 0])
        self._pending[key] = {
            "action": action,
            "old_log_prob": float(log_probs[action]),
            "value": float(value),
            "legal_mask": mask.copy(),
        }
        return action

    def store_episode(self, current_state, action, reward, next_state, done,
                      phase="play", actor_id="default", episode_id=0):
        """Commit the last sampled action for this actor/phase to the shared rollout."""
        phase = self._phase_name(phase)
        key = (str(actor_id), phase)
        info = self._pending.pop(key, None)
        if info is None or int(action) != info["action"]:
            # Includes forced-random recovery actions, which should not train PPO.
            return False
        state = np.asarray(current_state, dtype=np.float32).reshape(-1).copy()
        if state.size != self.state_size:
            raise ValueError(f"State has {state.size} features; expected {self.state_size}")
        nxt = None if next_state is None else np.asarray(next_state, dtype=np.float32).reshape(-1).copy()
        if nxt is not None and nxt.size != self.state_size:
            raise ValueError(f"Next state has {nxt.size} features; expected {self.state_size}")
        self.memory_buffer.append({
            "state": state, "action": int(action), "reward": float(reward),
            "next_state": nxt, "done": bool(done),
            "old_log_prob": info["old_log_prob"], "value": info["value"],
            "legal_mask": info["legal_mask"], "phase": phase,
            "actor_id": str(actor_id), "episode_id": int(episode_id),
        })
        if len(self.memory_buffer) > self.max_memory_buffer:
            del self.memory_buffer[:len(self.memory_buffer) - self.max_memory_buffer]
        return True

    def update_exploration_probability(self):
        # PPO exploration comes from sampling the categorical policy.
        return None

    def train(self):
        if not self.memory_buffer:
            print("No shared PPO samples; skipping update.")
            return
        rollout = self.memory_buffer
        states = np.stack([e["state"] for e in rollout]).astype(np.float32)
        actions = np.asarray([e["action"] for e in rollout], np.int32)
        old_log_probs = np.asarray([e["old_log_prob"] for e in rollout], np.float32)
        rewards = np.asarray([e["reward"] for e in rollout], np.float32)
        dones = np.asarray([e["done"] for e in rollout], np.float32)
        old_values = np.asarray([e["value"] for e in rollout], np.float32)
        phases = np.asarray([e["phase"] for e in rollout])
        masks = [e["legal_mask"] for e in rollout]

        # Bootstrap each transition independently, but never across terminal/phase
        # boundaries. GAE traces are grouped by actor + phase + game.
        next_values = np.zeros(len(rollout), np.float32)
        bootstrap_indices = [
            i for i, e in enumerate(rollout)
            if not e["done"] and e["next_state"] is not None
        ]
        if bootstrap_indices:
            nxt_states = np.stack([rollout[i]["next_state"] for i in bootstrap_indices])
            out = self._forward(nxt_states, training=False)
            vals = tf.squeeze(out["value"], axis=-1).numpy()
            for i, v in zip(bootstrap_indices, vals):
                next_values[i] = v

        advantages = np.zeros(len(rollout), np.float32)
        # Collection is interleaved across players. Sort each stream by insertion
        # order, then compute GAE backwards without mixing players or phases.
        streams = defaultdict(list)
        for i, e in enumerate(rollout):
            streams[(e["actor_id"], e["phase"], e["episode_id"])].append(i)
        for indices in streams.values():
            gae = 0.0
            for i in reversed(indices):
                nonterminal = 1.0 - dones[i]
                delta = rewards[i] + self.gamma * next_values[i] * nonterminal - old_values[i]
                gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
                advantages[i] = gae
        returns = advantages + old_values

        # Normalize globally for a stable joint update. Each batch may contain
        # both phases; the matching head/mask is selected per row below.
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(rollout))
        last_policy_loss = last_value_loss = last_entropy = 0.0
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_states = tf.convert_to_tensor(states[batch_idx])
                batch_actions = tf.convert_to_tensor(actions[batch_idx], tf.int32)
                batch_old_lp = tf.convert_to_tensor(old_log_probs[batch_idx])
                batch_adv = tf.convert_to_tensor(advantages[batch_idx])
                batch_returns = tf.convert_to_tensor(returns[batch_idx])
                batch_phases = phases[batch_idx]
                with tf.GradientTape() as tape:
                    output = self._forward(batch_states, training=True)
                    selected_lps, entropies = [], []
                    for j, original_idx in enumerate(batch_idx):
                        phase = rollout[original_idx]["phase"]
                        raw_logits = output["bid_logits" if phase == "bid" else "play_logits"][j]
                        mask = tf.convert_to_tensor(masks[original_idx], tf.bool)
                        masked_logits = tf.where(mask, raw_logits, tf.cast(-1e9, raw_logits.dtype))
                        log_probs = tf.nn.log_softmax(masked_logits)
                        selected_lps.append(tf.gather(log_probs, batch_actions[j]))
                        probs = tf.nn.softmax(masked_logits)
                        entropies.append(-tf.reduce_sum(probs * log_probs))
                    new_lp = tf.stack(selected_lps)
                    entropy = tf.reduce_mean(tf.stack(entropies))
                    values = tf.squeeze(output["value"], axis=-1)
                    ratio = tf.exp(new_lp - batch_old_lp)
                    clipped = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * batch_adv, clipped * batch_adv))
                    value_loss = tf.reduce_mean(tf.square(batch_returns - values))
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                grads = tape.gradient(loss, self.model.trainable_variables)
                pairs = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
                if pairs:
                    clipped_grads, _ = tf.clip_by_global_norm([g for g, _ in pairs], self.max_grad_norm)
                    self.optimizer.apply_gradients(zip(clipped_grads, [v for _, v in pairs]))
                last_policy_loss = float(policy_loss.numpy())
                last_value_loss = float(value_loss.numpy())
                last_entropy = float(entropy.numpy())

        self.model.save_weights(self.checkpoint_path)
        self.epoch += 1
        print(
            f"Shared PPO update {self.epoch}: samples={len(rollout)}, "
            f"mean_reward={rewards.mean():.3f}, policy_loss={last_policy_loss:.4f}, "
            f"value_loss={last_value_loss:.4f}, entropy={last_entropy:.4f}"
        )
        self.memory_buffer.clear()
        self._pending.clear()
