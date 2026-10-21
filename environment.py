"""Wizard websocket environment with fixed-card observations and shared PPO."""
import json
import threading
import websocket
import numpy as np
import sys
import traceback

from q_agent import QAgent


class WizardCallback:
    def on_turn(self, ws, state, players): pass
    def on_error(self, ws, message): pass
    def on_state_update(self, ws, state): pass
    def on_player_update(self, ws, players): pass
    def on_choosing_trump(self, ws, state, players): pass


class WizardGame:
    def __init__(self, username, callback, url="ws://localhost:6791"):
        self.username = username
        self.callback = callback
        self.players = None
        self.state = None
        self.ws_app = websocket.WebSocketApp(
            url, on_open=self.on_open, on_close=self.on_close,
            on_message=self.on_message, on_error=self.on_error)

    def get_turn(self):
        return any(p["name"] == self.username and p["turn"] for p in (self.players or []))

    def on_message(self, ws, message):
        data = json.loads(message)
        kind = data.get("type")
        if kind == "joined":
            print(f"{self.username}: joined game")
        elif kind == "state":
            self.state = data
            self.callback.on_state_update(ws, data)
            if data.get("choosing_trump") is not None:
                self.callback.on_choosing_trump(ws, data, self.players or [])
        elif kind == "player":
            self.players = data["players"]
            self.callback.on_player_update(ws, self.players)
            if self.get_turn() and self.state and self.state.get("choosing_trump") is None:
                self.callback.on_turn(ws, self.state, self.players)
        elif kind == "error":
            self.callback.on_error(ws, data.get("msg", "unknown server error"))
            if self.get_turn() and self.state and self.state.get("choosing_trump") is None:
                self.callback.on_turn(ws, self.state, self.players or [])

    def on_error(self, ws, error):
        print(f"{self.username}: websocket error: {error}")
        traceback.print_exc()

    def on_close(self, ws, *args): pass

    def on_open(self, ws):
        ws.send(json.dumps({"action": "join", "name": self.username}))

    def send(self, msg):
        self.ws_app.send(msg)

    def start(self):
        self.ws_app.run_forever()

    def close(self):
        try:
            self.ws_app.close()
        except Exception:
            pass


def _alloc(spec):
    offsets, cursor = {}, 0
    for name, size in spec:
        offsets[name] = cursor
        cursor += size
    return offsets, cursor


class WizardEnv(WizardCallback):
    """Per-player interface. All instances use the same shared policy/optimizer."""

    NUM_CARDS = 60
    NUM_PLAY_ACTIONS = 60
    NUM_BID_ACTIONS = 21
    # All observations have the same fixed length in both phases.
    OFFSETS, OBS_DIM = _alloc([
        ("round", 1), ("phase", 2), ("players", 1),
        ("trump", 5), ("lead", 6),
        ("tricks_won", 6), ("tricks_needed", 6), ("scores", 6), ("cards_to_play", 1),
        ("hand", NUM_CARDS), ("played", NUM_CARDS), ("table", NUM_CARDS), ("unseen", NUM_CARDS),
    ])

    def __init__(self, username, train=True, shared_agent=None, websocket_url="ws://localhost:6791"):
        self.username = username
        self.train = bool(train)
        self.agent = shared_agent or QAgent(self.OBS_DIM, env=self)
        self.websocket_url = websocket_url
        self.players = None
        self.state = None
        self.played = np.zeros(self.NUM_CARDS, dtype=bool)
        self.playing_reward = 0.0
        self.announcing_reward = 0.0
        self.last_playing_state = None
        self.last_playing_action = None
        self.last_announcing_state = None
        self.last_announcing_action = None
        self.game_index = 0
        self.previous_players = None
        self._lock = threading.RLock()
        self.game = None

    def play_game(self, game_index=None):
        if game_index is not None:
            self.game_index = int(game_index)
        self.correct_rounds = 0
        self.playing_reward = 0.0
        self.announcing_reward = 0.0
        self.players = None
        self.state = None
        self.played = np.zeros(self.NUM_CARDS, dtype=bool)
        self.previous_players = None
        self.last_playing_state = self.last_playing_action = None
        self.last_announcing_state = self.last_announcing_action = None
        self.game = WizardGame(self.username, self, self.websocket_url)
        self.game.start()

    def send(self, msg):
        if self.game:
            self.game.send(msg)

    @property
    def actor_id(self):
        return self.username

    def get_player(self, players):
        for i, player in enumerate(players or []):
            if player["name"] == self.username:
                return i, player
        raise ValueError(f"Player {self.username!r} not present in player list")

    def _finish_pending(self, phase, next_state, done):
        if not self.train:
            return
        if phase == "play" and self.last_playing_state is not None:
            self.agent.store_episode(
                self.last_playing_state, self.last_playing_action,
                self.playing_reward, next_state, done, phase="play",
                actor_id=self.actor_id, episode_id=self.game_index)
            self.last_playing_state = self.last_playing_action = None
            self.playing_reward = 0.0
        elif phase == "bid" and self.last_announcing_state is not None:
            self.agent.store_episode(
                self.last_announcing_state, self.last_announcing_action,
                self.announcing_reward, next_state, done, phase="bid",
                actor_id=self.actor_id, episode_id=self.game_index)
            self.last_announcing_state = self.last_announcing_action = None
            self.announcing_reward = 0.0

    def get_valid_announcing_actions(self, state, players):
        # state["round"] is the number of cards dealt this round; bids are 0..round.
        legal = list(range(max(0, min(int(state["round"]), 20)) + 1))

        # Sum of bids is not allowed to equal round (forcing a loser of the round)
        # -1 is used to indicate a player did not bid yet
        # Therefore if there's only one -1, we are placing the last bid
        announcements = [p["announcement"] for p in players]
        if announcements.count(-1) == 1:
            announcements.remove(-1)
            illegal = state["round"] - sum(announcements)
            if illegal >= 0:
                print(f"Round {state["round"]}, legal {legal}, illegal {illegal}")
                legal.remove(illegal)
        return legal

    def _card_action_id(self, card):
        return card["id"]

    def get_valid_playing_actions(self, state):
        # Build the same stable card IDs used by encode_cards(hand), then map the
        # selected ID back to a card object in on_turn. Number cards are unique;
        # Wizard/Fool occurrences are indexed in their order within the hand.
        legal = []
        i = 0
        lead = None
        while (lead is None or lead["type"] == "fool") and i < len(state["table"]):
            lead = state["table"][i]
            i += 1
        any_color = True
        if not lead is None and lead["type"] == "number":
            for card in state["hand"]:
                if "color" in card and card["color"] == lead["color"]:
                    any_color = False
                    break
        for card in state["hand"]:
            if card.get("type") == "wizard":
                legal.append(self._card_action_id(card))
            elif card.get("type") == "fool":
                legal.append(self._card_action_id(card))
            elif any_color or lead["color"] == card["color"]:
                legal.append(self._card_action_id(card))
        return legal

    def _card_for_action(self, hand, action):
        for card in hand:
            if card["id"] == int(action):
                return card
        raise ValueError(f"Chosen action {action} does not map to current hand")

    def on_turn(self, ws, state, players):
        enc_state = self.encode_state(state, players)
        is_bid = bool(state.get("announcing", False))

        # Commit the previous action of this same phase when its consequence is
        # observable. A phase boundary closes the previous phase's trajectory.
        if is_bid:
            self._finish_pending("play", None, True)
            if self.last_announcing_state is not None:
                self._finish_pending("bid", enc_state, False)
            legal = self.get_valid_announcing_actions(state, players)
            action = self.agent.compute_action(
                enc_state, legal,
                phase="bid", actor_id=self.actor_id)
            ws.send(json.dumps({"action": "announce", "announcement": action}))
            self.last_announcing_state, self.last_announcing_action = enc_state.copy(), action
        else:
            if self.last_playing_state is not None:
                self._finish_pending("play", enc_state, False)
            legal = self.get_valid_playing_actions(state)
            action = self.agent.compute_action(
                enc_state, legal,
                phase="play", actor_id=self.actor_id)
            card = self._card_for_action(state["hand"], action)
            ws.send(json.dumps({"action": "play_card", **card}))
            self.last_playing_state, self.last_playing_action = enc_state.copy(), action

    def on_choosing_trump(self, ws, state, players):
        # Keep this deterministic baseline behavior; trump choice is not yet an
        # action head. It can be added as a third head if the server exposes it.
        if state.get("choosing_trump") == self.username:
            ws.send(json.dumps({"action": "choose_trump", "color": "red"}))

    def on_state_update(self, ws, state):
        self.state = state
        if state.get("game_over"):
            # Finalize any outstanding sampled action as terminal.
            self._finish_pending("play", None, True)
            self._finish_pending("bid", None, True)
            print(f"Game finished with scores {[p["score"] for p in self.players]}, correct bids: {self.correct_rounds}")
            ws.close()

    def on_player_update(self, ws, players):
        if self.previous_players:
            _, old = self.get_player(self.previous_players)
            _, new = self.get_player(players)
            if old.get("tricks") != new.get("tricks"):
                # Dense, small shaping reward; exact round score is also
                # assigned to the bid decision below.
                self.playing_reward += -10.0 if new["tricks"] > new["announcement"] else 10.0
                #if new["tricks"] == new["announcement"]:
                #    self.playing_reward += 5.0
            if old.get("score") != new.get("score"):
                # Higher rounds score higher, therefore normalize to the round number
                self.announcing_reward += (new["score"] - old["score"])# / self.state["round"]
                # Don't penalize playing head to strongly in case of a bad bid
                self.playing_reward += 10.0 if new["score"] > old["score"] else -10.0
                self.correct_rounds += 1 if new["score"] > old["score"] else 0
        self.previous_players = [dict(p) for p in players]
        self.players = players

    def on_error(self, ws, message):
        # Websocket errors can be emitted for invalid moves; next decision is
        # selected randomly from legal moves and excluded from PPO training.
        print(f"{self.username}: server error: {message}")

    def encode_state(self, state, players):
        encoded = np.zeros(self.OBS_DIM, dtype=np.float32)
        off = self.OFFSETS
        n_rounds = int(self.NUM_CARDS / len(players))
        encoded[off["round"]] = state["round"] / n_rounds
        encoded[off["players"]] = len(players)
        encoded[off["phase"] + (1 if state.get("announcing") else 0)] = 1.0

        trump = state.get("trump")
        if trump is None or trump.get("type") == "fool":
            encoded[off["trump"] + 4] = 1.0
        else:
            encoded[off["trump"] + self.stc(trump["color"])] = 1.0

        table = state.get("table")
        i = 0
        lead = None
        while (lead is None or lead["type"] == "fool") and i < len(state["table"]):
            lead = state["table"][i]
            i += 1
        if lead is None:
            pass  # all-zero lead vector means no card has led this trick yet
        elif lead.get("type") == "wizard":
            encoded[off["lead"] + 4] = 1.0
        elif lead.get("type") == "fool":
            encoded[off["lead"] + 5] = 1.0
        else:
            encoded[off["lead"] + self.stc(lead["color"])] = 1.0

        encoded[off["cards_to_play"]] = len(state["hand"]) / state["round"]

        # Rotate player features so slot 0 always represents the acting player.
        actor_index, _ = self.get_player(players)
        relative_players = list(players[actor_index:]) + list(players[:actor_index])
        for i, p in enumerate(relative_players):
            encoded[off["scores"] + i] = p["score"]
            encoded[off["tricks_won"] + i] = p["tricks"] / state["round"]
            announced = p["announcement"]
            if announced != -1:
                encoded[off["tricks_needed"] + i] = (announced - p["tricks"]) / state["round"]

        encoded_hand = self.encode_cards(state["hand"])
        encoded_table =  self.encode_cards(table)
        encoded[off["hand"]:off["hand"] + self.NUM_CARDS] = encoded_hand
        encoded[off["table"]:off["table"] + self.NUM_CARDS] = encoded_table

        self.played |= np.astype(encoded_table, bool)
        if trump:
            self.played[self._card_action_id(trump)] = True

        encoded[off["played"]:off["played"] + self.NUM_CARDS] = self.played
        encoded[off["unseen"]:off["unseen"] + self.NUM_CARDS] = ~(self.played |
                                                                  np.astype(encoded_hand, bool))
        return encoded

    def encode_cards(self, cards):
        deck = np.zeros(self.NUM_CARDS, dtype=np.float32)
        wizard_n = fool_n = 0
        for card in cards:
            if not card:
                continue
            typ = card.get("type")
            if typ == "number":
                deck[13 * self.stc(card["color"]) + int(card["number"]) - 1] = 1.0
            elif typ == "wizard":
                if wizard_n < 4:
                    deck[52 + wizard_n] = 1.0
                    wizard_n += 1
            elif typ == "fool":
                if fool_n < 4:
                    deck[56 + fool_n] = 1.0
                    fool_n += 1
        return deck

    @staticmethod
    def stc(color):
        return ["red", "blue", "green", "yellow"].index(color)

    def train_agents(self):
        # Called once by train.py after every player has finished the game.
        # Close any final decisions before the one shared optimizer update.
        self._finish_pending("play", None, True)
        self._finish_pending("bid", None, True)
