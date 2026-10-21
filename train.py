"""Run shared-policy self-play against the local Wizard websocket server.

Usage: python train.py NUM_PLAYERS NUM_GAMES
The game server must be running on ws://localhost:6791.
"""
import sys
import time
from threading import Thread

from environment import WizardEnv
from q_agent import QAgent


def main(num_players, num_games):
    state_size = WizardEnv.OBS_DIM
    shared_agent = QAgent(state_size)
    shared_agent.load_weights()

    players = [
        WizardEnv(f"KI {i}", train=True, shared_agent=shared_agent)
        for i in range(num_players)
    ]

    for game_idx in range(num_games):
        threads = []
        for player in players:
            thread = Thread(target=player.play_game, args=(game_idx,), daemon=True)
            thread.start()
            threads.append(thread)
            time.sleep(0.1)

        # Keep the original server protocol: start after clients have joined.
        time.sleep(1.0)
        players[0].send('{"action": "start_game"}')
        for thread in threads:
            thread.join()

        # Every player has contributed to one shared rollout; update once.
        shared_agent.train()
        print(f"Finished game {game_idx + 1}/{num_games}")

    print("Training complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python train.py NUM_PLAYERS NUM_GAMES")
    main(int(sys.argv[1]), int(sys.argv[2]))
