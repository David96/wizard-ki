"""Run shared-policy self-play against the local Wizard websocket server.

Usage: python train.py NUM_PLAYERS NUM_GAMES [TRAIN [N_GAMES [FROZEN_CHECKPOINT]]]
The game server must be running on ws://localhost:6791 (+ one instance per
extra pooled game, see N_GAMES).
"""
import sys
import time
from threading import Thread

from environment import WizardEnv
from actor_critic import ActorCriticAgent


def main(num_players, num_games, train=True, pooled_games=6,
          frozen_checkpoint=None):
    state_size = WizardEnv.OBS_DIM
    print(f"Input state size: {state_size}")
    shared_agent = ActorCriticAgent(state_size, train)
    shared_agent.load_weights()
    frozen_agent = None
    if frozen_checkpoint:
        # Past-self opponent: one seat per pooled game plays greedily from an
        # old checkpoint and contributes no rollouts. Keeps the shared policy
        # from overfitting to its own current bid distribution and doubles as
        # a progress metric (win rate against the frozen reference).
        frozen_agent = ActorCriticAgent(state_size, False,
                                        checkpoint_path=frozen_checkpoint)
        frozen_agent.load_weights()
        print(f"Frozen opponent on one seat per game: {frozen_checkpoint}")

    port = 6791
    players = []
    for k in range(pooled_games):
        for i in range(num_players):
            use_frozen = frozen_agent is not None and i == num_players - 1
            players.append(WizardEnv(
                f"KI {i+k*num_players}", train and not use_frozen,
                frozen_agent if use_frozen else shared_agent,
                websocket_url=f"ws://localhost:{port}"))
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
        for i in range(0, pooled_games * num_players, num_players):
            players[i].send('{"action": "start_game"}')
        for thread in threads:
            thread.join()

        # Close any remaining decisions before the one shared optimizer update;
        # otherwise events without a terminal done flag would be dropped.
        for player in players:
            player.train_agents()
        shared_agent.train()
        print(f"Finished game {game_idx + 1}/{num_games}")

    print("Training complete.")


if __name__ == "__main__":
    if not 3 <= len(sys.argv) <= 6:
        raise SystemExit("Usage: python train.py NUM_PLAYERS NUM_GAMES "
                         "[TRAIN [N_GAMES [FROZEN_CHECKPOINT]]]")
    train = True
    if len(sys.argv) >= 4:
        # Only treat explicit truthy strings as training; bool("False") would
        # silently enable training for every non-empty argument.
        train = sys.argv[3].lower() in ("1", "true", "yes")
    pooled_games = 6
    if len(sys.argv) >= 5:
        # Number of games collected in parallel under frozen weights that are
        # pooled into a single PPO update; each runs on its own server port.
        pooled_games = int(sys.argv[4])
    frozen_checkpoint = None
    if len(sys.argv) == 6:
        # Optional checkpoint of an older policy; if given, the last seat of
        # every pooled game plays against it without contributing rollouts.
        frozen_checkpoint = sys.argv[5]
    main(int(sys.argv[1]), int(sys.argv[2]), train, pooled_games,
         frozen_checkpoint)
