"""Run shared-policy self-play against the local Wizard websocket server.

Usage: python train.py NUM_PLAYERS NUM_GAMES [TRAIN [N_GAMES [FROZEN_CHECKPOINT]]]
The game server must be running on ws://localhost:6791. Each pooled game
creates its own room on the server (which supports up to 20 at a time).
"""
import sys
import time
from threading import Thread

from environment import WizardEnv, RoomGroup
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

    players = []
    for k in range(pooled_games):
        # One room per pooled game: the first client creates it (and becomes
        # the room creator, i.e. the one allowed to send start_game), the
        # others wait for the published room id and join.
        group = RoomGroup()
        for i in range(num_players):
            use_frozen = frozen_agent is not None and i == num_players - 1
            players.append(WizardEnv(
                f"KI {i+k*num_players}", train and not use_frozen,
                frozen_agent if use_frozen else shared_agent,
                room_group=group, create_room=(i == 0), close_delay=i))

    for game_idx in range(num_games):
        threads = []
        for player in players:
            thread = Thread(target=player.play_game, daemon=True)
            thread.start()
            threads.append(thread)
            time.sleep(0.1)

        # Give every client time to join its room, then start all pooled
        # games. The first player of each game is the room creator and the
        # only one the server lets send start_game.
        time.sleep(1.0)
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
