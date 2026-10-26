"""Run shared-policy self-play against the local Wizard websocket server.

Usage: python train.py NUM_PLAYERS NUM_GAMES
The game server must be running on ws://localhost:6791.
"""
import sys
import time
from threading import Thread

from environment import WizardEnv
from actor_critic import ActorCriticAgent


def main(num_players, num_games):
    state_size = WizardEnv.OBS_DIM
    shared_agent = ActorCriticAgent(state_size)
    shared_agent.load_weights()

    n_games = 6
    port = 6791
    players = []
    for _ in range(n_games):
        players += [
            WizardEnv(f"KI {i}", True, shared_agent,
                      websocket_url=f"ws://localhost:{port}")
            for i in range(num_players)
        ]
        port += 1

    for game_idx in range(num_games):
        threads = []
        for player in players:
            thread = Thread(target=player.play_game, daemon=True)
            thread.start()
            threads.append(thread)
            time.sleep(0.1)

        # Keep the original server protocol: start after clients have joined.
        time.sleep(1.0)
        # The first player in a game is the game master, use them to start the game
        for i in range(0, n_games * num_players, num_players):
            players[i].send('{"action": "start_game"}')
        for thread in threads:
            thread.join()

        shared_agent.train()
        print(f"Finished game {game_idx + 1}/{num_games}")

    print("Training complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python train.py NUM_PLAYERS NUM_GAMES")
    main(int(sys.argv[1]), int(sys.argv[2]))
