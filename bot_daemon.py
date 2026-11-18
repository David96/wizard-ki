#!/bin/python
"""Single-process bot daemon for the wizard game server.

Runs any number of bots inside one process: every bot gets its own
websocket connection and thread, but they all share a single loaded
ActorCriticAgent. Memory stays flat no matter how many bots play:
one TensorFlow plus one model instead of one copy per bot process.

Control protocol: newline-delimited JSON over TCP on 127.0.0.1, default
port one above the game server's port. A single persistent control
connection is expected; every command line is answered with one reply
line, and the connection doubles as a liveness lease:

    {"cmd": "add", "room": "AB12CD", "name": "BertieBot"}  -> {"ok": true}
    {"cmd": "stop", "name": "BertieBot", "room": "AB12CD"} -> {"ok": true, "stopped": 1}
    {"cmd": "stop_room", "room": "AB12CD"}                 -> {"ok": true, "stopped": 3}
    {"cmd": "ping"}                                        -> {"ok": true, "bots": 0, "model": false}
    {"cmd": "shutdown"}                                    -> {"ok": true}

Errors reply {"ok": false, "error": "..."}. "add" takes an optional
"delay" (think delay) per bot; the default comes from --think-delay.
The first "add" pays the TensorFlow import and model load, so its reply
takes a few seconds; it is sent only once the model is ready, which
also signals whether the bot could start at all.

"stop"/"stop_room" are only valid once the bot's game is running:
in the brief window between "add" and the worker entering play_game()
there is no game to close yet, so those bots are reported as failures
(ok false, "stopped" counts the bots that were closed) and the caller
may retry a moment later. This window is tiny and rarely hit.

Lifecycle: TensorFlow is imported and the model loaded lazily on the
first "add", so an idle daemon costs almost nothing. When the control
connection drops and no bots are running, the daemon exits after
--linger seconds (freeing the model memory); a new control connection
or a running bot cancels the exit. The games server's BotManager is
expected to spawn this daemon lazily and reconnect.
"""

from argparse import ArgumentParser

import json
import os
import socket
import threading
import time
import traceback
from urllib.parse import urlparse

DEFAULT_URL = "ws://127.0.0.1:6791"
DEFAULT_CHECKPOINT = "training_actor_critic/shared_policy.weights.h5"
DEFAULT_THINK_DELAY = 0.8
DEFAULT_MAX_BOTS = 8
DEFAULT_LINGER = 60.0


def log(msg):
    print(f"[daemon] {msg}", flush=True)


class BotRecord:
    """One running bot: its room, name and the env built by the control
    thread before the worker starts. The game itself is created by
    play_game() inside the worker."""

    def __init__(self, room, name, env):
        self.room = room
        self.name = name
        self.env = env
        self.done = threading.Event()

    def stop(self):
        """Close the bot's game, making play_game() return in the worker.
        Returns False if the game has not started yet: stopping in the
        brief window between add and the worker entering play_game() is
        invalid; the caller may retry."""
        game = self.env.game
        if game is None:
            return False
        game.close()
        return True


class BotDaemon:
    def __init__(self, url, checkpoint, delay, bind, port, linger, max_bots):
        self.url = url
        self.checkpoint = checkpoint
        self.delay = delay
        self.bind = bind
        self.port = port
        self.linger = linger
        self.max_bots = max_bots
        self.bots = {}                    # (room, name) -> BotRecord
        self.lock = threading.Lock()
        self.agent = None
        self.agent_error = None
        self._agent_lock = threading.Lock()
        self._WizardEnv = None
        self._RoomGroup = None
        self._stop = False
        self._ever_connected = False
        self._idle_since = None

    # -- model -----------------------------------------------------------

    def _get_agent(self):
        """Load the shared agent once, on the first add.

        Imports are deferred so an idle daemon does not hold TensorFlow
        in memory. Inference (compute_action with training=False) is
        read-only and safe to call from several bot threads at once.
        """
        with self._agent_lock:
            if self.agent is None and self.agent_error is None:
                try:
                    log("importing tensorflow and loading the shared model...")
                    # Imported here, not at the top, see docstring
                    from actor_critic import ActorCriticAgent
                    from environment import RoomGroup, WizardEnv
                    self._WizardEnv = WizardEnv
                    self._RoomGroup = RoomGroup
                    if not os.path.isfile(self.checkpoint):
                        log(f"warning: checkpoint {self.checkpoint!r} not found, "
                            "the bots will play random moves")
                    agent = ActorCriticAgent(WizardEnv.OBS_DIM, False,
                                             checkpoint_path=self.checkpoint)
                    agent.load_weights()
                    self.agent = agent
                    log("model ready")
                except Exception:
                    self.agent_error = "model load failed"
                    traceback.print_exc()
            return self.agent

    # -- bots ------------------------------------------------------------

    def _bot_worker(self, record):
        try:
            log(f"{record.name}: joining room {record.room}")
            record.env.play_game()   # blocks until the websocket closes
            log(f"{record.name}: game over")
        except Exception:
            traceback.print_exc()
        finally:
            record.done.set()
            with self.lock:
                self.bots.pop((record.room, record.name), None)
                if not self.bots:
                    self._mark_idle_locked()

    def add_bot(self, room, name, delay=None):
        """Start a bot; returns an error string or None on success."""
        agent = self._get_agent()
        if agent is None:
            return self.agent_error
        with self.lock:
            if (room, name) in self.bots:
                return f"bot {name!r} already runs in room {room!r}"
            if len(self.bots) >= self.max_bots:
                return f"bot limit of {self.max_bots} reached"
        rg = self._RoomGroup()
        rg.publish(0, room)
        env = self._WizardEnv(name, False, agent, websocket_url=self.url,
                              room_group=rg, create_room=False,
                              close_delay=0,
                              think_delay=self.delay if delay is None else delay)
        record = BotRecord(room, name, env)
        with self.lock:
            self.bots[(room, name)] = record
            self._idle_since = None
        threading.Thread(target=self._bot_worker, args=(record,),
                         daemon=True, name=f"bot-{name}").start()
        return None

    def _stop_matching(self, room=None, name=None):
        """Stop all matching bots. Returns (stopped, failed) with the
        records whose game was closed and those that could not be
        stopped because their game has not started yet."""
        with self.lock:
            records = [r for (r_room, r_name), r in self.bots.items()
                       if room in (None, r_room) and name in (None, r_name)]
        stopped, failed = [], []
        for record in records:
            (stopped if record.stop() else failed).append(record)
        # The stopped workers need a moment to leave play_game() after
        # their socket closed (a callback may still be in flight); wait
        # so that a "stopped" reply really means the bots are gone
        for record in stopped:
            record.done.wait(5.0)
        return stopped, failed

    def stop_all(self):
        self._stop_matching()

    # -- lifecycle ---------------------------------------------------------

    def _mark_idle_locked(self):
        """Start the linger clock once no bots are left.
        The caller must hold self.lock."""
        if self._ever_connected and not self.bots:
            self._idle_since = time.time()

    def _idle_expired(self):
        if not self._ever_connected or self.linger < 0:
            return False
        with self.lock:
            return (self._idle_since is not None and not self.bots
                    and time.time() - self._idle_since >= max(self.linger, 0.0))

    # -- control server ------------------------------------------------------

    def _handle(self, command):
        cmd = command.get("cmd")
        if cmd == "ping":
            with self.lock:
                bots = len(self.bots)
            return {"ok": True, "bots": bots, "model": self.agent is not None}
        if cmd == "add":
            room, name = command.get("room"), command.get("name")
            if not room or not name:
                return {"ok": False, "error": "add needs 'room' and 'name'"}
            error = self.add_bot(room, name, command.get("delay"))
            if error:
                return {"ok": False, "error": error}
            return {"ok": True}
        if cmd == "stop":
            if not command.get("name"):
                return {"ok": False, "error": "stop needs 'name'"}
            return self._stop_reply(self._stop_matching(
                room=command.get("room"), name=command["name"]))
        if cmd == "stop_room":
            if not command.get("room"):
                return {"ok": False, "error": "stop_room needs 'room'"}
            return self._stop_reply(self._stop_matching(room=command["room"]))
        if cmd == "shutdown":
            self._stop = True
            return {"ok": True}
        return {"ok": False, "error": f"unknown command {cmd!r}"}

    @staticmethod
    def _stop_reply(result):
        stopped, failed = result
        reply = {"ok": not failed, "stopped": len(stopped)}
        if failed:
            reply["error"] = (", ".join(r.name for r in failed)
                              + " not stopped: their game has not started yet "
                                "(invalid this early after add; retry)")
        return reply

    def _serve_control(self, conn):
        reader = conn.makefile("r", encoding="utf-8")
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                reply = self._handle(json.loads(line))
            except json.JSONDecodeError:
                reply = {"ok": False, "error": "malformed json"}
            log(f"control: {line} -> {reply['ok']}")
            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

    def run(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind, self.port))
        listener.listen(1)
        listener.settimeout(0.5)
        log(f"control socket on {self.bind}:{self.port}, game server {self.url}")
        try:
            while not self._stop:
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    if self._idle_expired():
                        log("idle: no control connection and no bots - exiting")
                        return
                    continue
                self._ever_connected = True
                self._idle_since = None
                log("control connection established")
                try:
                    self._serve_control(conn)
                except OSError:
                    log("control connection dropped abruptly")
                finally:
                    conn.close()
                    log("control connection closed")
                    with self.lock:
                        self._mark_idle_locked()
        finally:
            listener.close()
            self.stop_all()


def main():
    p = ArgumentParser(description="Single-process wizard bot daemon")
    p.add_argument('-u', '--url', default=DEFAULT_URL,
                   help=f"game server websocket url (default: {DEFAULT_URL})")
    p.add_argument('-c', '--checkpoint', default=DEFAULT_CHECKPOINT,
                   help=f"shared policy checkpoint (default: {DEFAULT_CHECKPOINT})")
    p.add_argument('-d', '--think-delay', type=float, default=DEFAULT_THINK_DELAY,
                   help=f"bot think delay in seconds (default: {DEFAULT_THINK_DELAY})")
    p.add_argument('-b', '--bind', default='127.0.0.1',
                   help="control socket bind address (default: 127.0.0.1)")
    p.add_argument('-p', '--port', type=int, default=None,
                   help="control port (default: game port + 1)")
    p.add_argument('--linger', type=float, default=DEFAULT_LINGER,
                   help="seconds to stay alive without a control connection "
                        "and without bots; 0 = exit immediately, -1 = never "
                        f"(default: {DEFAULT_LINGER})")
    p.add_argument('-m', '--max-bots', type=int, default=DEFAULT_MAX_BOTS,
                   help=f"concurrent bot cap (default: {DEFAULT_MAX_BOTS})")
    args = p.parse_args()
    if args.port is None:
        args.port = (urlparse(args.url).port or 80) + 1
    daemon = BotDaemon(args.url, args.checkpoint, args.think_delay,
                       args.bind, args.port, args.linger, args.max_bots)
    try:
        daemon.run()
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop_all()
        log("bye")


if __name__ == "__main__":
    main()
