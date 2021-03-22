import asyncio
import json
import _thread
import threading
import time
import websocket
import random

import numpy as np

from q_agent import QAgent

class WizardCallback:
    def on_turn(self, ws, state, players):
        pass
    def on_error(self, ws, message):
        pass
    def on_state_update(self, ws, state):
        pass
    def on_player_update(self, ws, players):
        pass
    def on_choosing_trump(self, ws, state, players):
        pass

class WizardGame:
    def get_turn(self):
        if self.players:
            for player in self.players:
                if player["name"] == self.username:
                    return player["turn"]
        return False

    def on_message(self, ws, message):
        data = json.loads(message)
        if data["type"] == "joined":
            print("Congrats, you joined")
        elif data["type"] == "state":
            self.state = data
            self.callback.on_state_update(ws, data)
            if data["choosing_trump"] is not None:
                self.callback.on_choosing_trump(ws, data, self.players)
        elif data["type"] == "player":
            self.players = data["players"]
            self.callback.on_player_update(ws, data["players"])
            if self.get_turn() and self.state["choosing_trump"] is None:
                self.callback.on_turn(ws, self.state, self.players)
        elif data["type"] == "error":
            self.callback.on_error(ws, data["msg"])
            # if we get an error we probably have to try again...
            if self.get_turn() and self.state["choosing_trump"] is None:
                self.callback.on_turn(ws, self.state, self.players)
            print("Fuck: " + data["msg"])

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        pass

    def on_open(self, ws):
        ws.send(json.dumps({"action": "join", "name": self.username}))

    def send(self, msg):
        self.ws_app.send(msg)

    def __init__(self, username, callback):
        self.username = username
        self.callback = callback
        self.ws_app = websocket.WebSocketApp("ws://localhost:6791",
                on_open=self.on_open, on_close=self.on_close,
                on_message=self.on_message, on_error=self.on_error)
        #websocket.enableTrace(True)

    def start(self):
        self.ws_app.run_forever()
        print("Forever is over")


class WizardEnv(WizardCallback):
    '''
    state layout playing:
                int
          current round [1,20]
                int
          number of players
                int[6]
          announcements
                int[20 * 2]
          hand cards: 1 int for strength, one for color
                int[6 * 2]
          table cards

          => state_size = 60

    state layout announcing:
                int
          current round [1,20]
                int
          number of players
                int[6]
          announcements
                int[20 * 2]
          hand cards: 1 int for strength, one for color

          => state_size = 48

    card layout:
        1st int: strength: 1-13 for normal card, 14-26 for trump, 27 wizard, 1 fool
        2nd int: color: 1-4 for normal colors, 0 for fool and wizard
    '''
    def __init__(self, username, train):
        self.state_size_playing = 60
        self.action_size_playing = 20
        self.state_size_announcing = 48
        self.action_size_announcing = 21
        self.player_count = 6
        self.username = username
        self.playing_reward = 0
        self.announcing_reward = 0
        self.players = None
        self.state = None
        self.last_playing_state = None
        self.last_playing_action = None
        self.last_announcing_state = None
        self.last_announcing_action = None
        self.wrong_move = False
        self.train = train
        self.agent_playing = QAgent(self.state_size_playing, self.action_size_playing)
        self.agent_announcing = QAgent(self.state_size_announcing, self.action_size_announcing)

    def play_game(self):
        self.playing_reward = 0
        self.announcing_reward = 0
        self.players = None
        self.state = None
        self.last_playing_state = None
        self.last_playing_action = None
        self.last_announcing_state = None
        self.last_announcing_action = None
        for agent in [self.agent_announcing, self.agent_playing]:
            agent.memory_buffer = list()
            agent.load_weights()
        self.game = WizardGame(self.username, self)
        self.game.start()

    def send(self, msg):
        self.game.send(msg)

    def on_turn(self, ws, state, players):
        enc_state = self.encode_state(state, players)
        if state["announcing"]:
            if self.train and self.last_announcing_state is not None:
                self.agent_announcing.store_episode(self.last_announcing_state, self.last_announcing_action,
                        self.announcing_reward, enc_state, False)
            action_space = list(range(state["round"] + 1))
            force_random = False
            if self.wrong_move:
                force_random = True
            elif self.train and self.last_playing_state is not None:
                self.agent_playing.store_episode(self.last_playing_state, self.last_playing_action,
                        self.playing_reward, None, True)
                self.last_playing_state = None
            action = self.agent_announcing.compute_action(enc_state, action_space, force_random)
            ws.send(json.dumps({"action":"announce", "announcement":action}))

            self.announcing_reward = self.playing_reward = 0
            self.last_announcing_action = action
            self.last_announcing_state = enc_state
        else:
            if self.train and self.last_playing_state is not None:
                self.agent_playing.store_episode(self.last_playing_state, self.last_playing_action,
                        self.playing_reward, enc_state, False)
            action_space = list(range(len(state["hand"])))
            if self.wrong_move:
                action = self.agent_playing.compute_action(enc_state, action_space, True)
            else:
                action = self.agent_playing.compute_action(enc_state, action_space)
            card = state["hand"][action]
            ws.send(json.dumps({"action":"play_card", **card}))
            self.playing_reward = 0
            self.last_playing_action = action
            self.last_playing_state = enc_state
        self.wrong_move = False

    def on_choosing_trump(self, ws, state, players):
        if state["choosing_trump"] == self.username:
            ws.send(json.dumps({"action": "choose_trump", "color": "red"}))

    def on_state_update(self, ws, state):
        self.state = state
        if state["game_over"]:
            if self.train:
                self.agent_playing.store_episode(self.last_playing_state, self.last_playing_action,
                        self.playing_reward, None, True)
                self.agent_announcing.store_episode(self.last_announcing_state, self.last_announcing_action,
                        self.announcing_reward, None, True)
                self.agent_playing.train()
                self.agent_announcing.train()
            self.agent_playing.update_exploration_probability()
            self.agent_announcing.update_exploration_probability()
            ws.close()

    def on_player_update(self, ws, players):
        if self.players:
            old_player = self.get_player(self.players)
            new_player = self.get_player(players)
            if old_player["tricks"] != new_player["tricks"]:
                self.playing_reward += -10 if new_player["tricks"] > new_player["announcement"] else 5
                if new_player["tricks"] == new_player["announcement"]:
                    self.playing_reward += 15
            if old_player["score"] != new_player["score"]:
                self.announcing_reward += new_player["score"] - old_player["score"]
        self.players = players

    def on_error(self, ws, msg):
        if msg != 'It\'s not your turn, bitch':
            if msg != 'Nope. Wrong number ¯\\_(ツ)_/¯':
                self.playing_reward -= 10
            else:
                self.announcing_reward -= 10
            self.wrong_move = True

    def encode_state(self, state, players):
        encoded_state = np.zeros(self.state_size_announcing if state["announcing"] else self.state_size_playing)
        encoded_state[0] = state["round"]
        encoded_state[1] = len(players)
        for (i, p) in enumerate(players):
            encoded_state[i + 2] = p["announcement"]
        for (i, c) in enumerate(state["hand"]):
            nr, color = self.encode_card(c, state["trump"])
            encoded_state[3 + 6 + 2*i] = nr
            encoded_state[3 + 6 + 2*i + 1] = color
        if not state["announcing"]:
            for (i, c) in enumerate(state["table"]):
                nr, color = self.encode_card(c, state["trump"])
                encoded_state[3 + 6 + 20 + 2*i] = nr
                encoded_state[3 + 6 + 20 + 2*i + 1] = color
        return encoded_state

    def get_player(self, players):
        for p in players:
            if p["name"] == self.username:
                return p
        return None

    def encode_card(self, c, trump):
        if c["type"] == "wizard":
            nr = 27
            color = 0
        elif c["type"] == "fool":
            nr = 1
            color = 0
        elif c["type"] == "number":
            nr = c["number"]
            if trump and trump["type"] == "number" and c["color"] == trump["color"]:
                nr += 13
            color = self.stc(c["color"])
        return (nr, color)


    def stc(self, color):
        colors = ['red', 'blue', 'green', 'yellow', 'orange']
        return colors.index(color) + 1

