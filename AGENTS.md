# AGENTS.md

RL project: a shared PPO actor-critic that plays the card game Wizard by self-playing against an external Wizard websocket game server.

## Running training

- The game server is a separate repo at `/home/iuser/Dokumente/Projekte/games` (remote: `github.com/David96/games`). Start it from that repo's root: `python main.py [port]` (default port 6791, binds localhost only; only dependency is `websockets`). Game logic lives in `games/wizard/wizard.py`, room framework in `games/game_room.py`. The KI client code predates the server's room system:
  - `environment.py` sends `{"action": "join", "name": ...}` with no `room` field, but the current server requires `create_room` first and a `room` ID on `join` — joining as-is fails with `Invalid room ID!`.
  - `train.py`'s first KI client sends `start_game` to auto-start each self-play game (this is commented out when playing against the AI manually via the browser UI; against the room-based server the creator/start flow differs again).
  - Either adapt the client to the room protocol or use a matching older server version.
- `python train.py NUM_PLAYERS NUM_GAMES [TRAIN]`
  - The third arg goes through `bool(sys.argv[3])`, so any non-empty string (including `False`) evaluates to training mode. There is currently no way to disable training from the CLI.
  - `n_games` (hardcoded in `train.py`, currently set to 1) is the number of parallel games per update step, not the training length: each game gets its own server port starting at 6791 (`port += 1`), so `n_games = 6` needs server instances on ports 6791–6796. All `n_games × NUM_PLAYERS` clients play concurrently under frozen weights and their rollouts are pooled into **one** PPO update. Training runs so far used `n_games = 6` — that's why `training_data.txt` has ~1 row per 6 games. Scale the update batch via `n_games`, not `NUM_GAMES`.
- `NUM_GAMES` is the number of pooled update steps: the loop reuses the same player objects; each `play_game()` opens a fresh websocket connection, then `shared_agent.train()` runs once per step.

## Architecture / file map

- `train.py` — entry point. Creates one shared `ActorCriticAgent` and one `WizardEnv` (websocket client) per player, each in a thread.
- `actor_critic.py` — `ActorCriticAgent`: the **active** shared PPO policy. Shared trunk + value head, separate 21-action bid head and 60-action play head. Checkpoints in `training_actor_critic/shared_policy.weights.h5`.
- `environment.py` — `WizardEnv`: websocket protocol handling, fixed 327-dim state encoding, legal-action computation, reward shaping.
- `q_agent.py` — abandoned earlier PPO draft (kept for reference only; name survived the DQN era). Do not use, extend, or load its checkpoints.
- Checkpoints are incompatible across implementations (`actor_critic.py` vs `q_agent.py` vs old DQN) — do not mix them.
- Versioned snapshots like `training_actor_critic/shared_policy-v2-24000.weights.h5` are made by manually copying the checkpoint. Code only ever reads/writes the unversioned `shared_policy.weights.h5`.

## Conventions & gotchas

- `training_data.txt` (losses per update) and `game_data.txt` (bids/tricks per game) are append-only training logs written by the code. Numbered variants (`-v2-36000`, `.old`, `-didntwork`) are manually archived logs. `notebooks/plot_losses.ipynb` and `notebooks/plot_progress.ipynb` plot these files.
- `ActorCriticAgent.phase1` switches reward semantics everywhere: `True` = bid head trained via cross-entropy against tricks won + greedy trick rewards; `False` (current) = PPO with score-delta bid rewards. Both `environment.py` (`on_player_update`) and `actor_critic.py` (`train`) branch on it — keep them consistent.
- Card actions are the server's card IDs (0–59 fixed deck slots), not hand indices. Bid actions are `0..round`; the bid that would make the announced total equal the round is removed for the last bidder.
- Trump choice is not learned: hardcoded to `"red"` in `WizardEnv.on_choosing_trump`.
- `training_*/` and `__pycache__` are gitignored; checkpoints never go to git.
- Active branch is `wip` (ahead of `master`); remote is `github.com/David96/wizard-ki`.

## Dependencies

- No requirements file exists. Imports: `tensorflow` (Keras 3 style, incl. `from keras import ops`), `websocket-client`, `numpy`. Pick a TensorFlow version compatible with your Python version.

## Reference docs

- `README.md` describes the `q_agent.py` shared-PPO rewrite design plus integration caveats (server state schema, reward/GAE boundaries, websocket lifecycle) that also apply to the active code.
- `README_ppo_integration.md` documents how `QAgent` replaced the old DQN agent and which assumptions to re-verify.
