# AGENTS.md

RL project: a shared PPO actor-critic that plays the card game Wizard by self-playing against an external Wizard websocket game server.

## Running training

- The game server is a separate repo at `/home/iuser/Dokumente/Projekte/games` (remote: `github.com/David96/games`). Start it from that repo's root: `python main.py [port]` (default port 6791, binds localhost only; only dependency is `websockets`). Game logic lives in `games/wizard/wizard.py`, room framework in `games/game_room.py`. The KI client code predates the server's room system:
  - `environment.py` sends `{"action": "join", "name": ...}` with no `room` field, but the current server requires `create_room` first and a `room` ID on `join` — joining as-is fails with `Invalid room ID!`.
  - `train.py`'s first KI client sends `start_game` to auto-start each self-play game (this is commented out when playing against the AI manually via the browser UI; against the room-based server the creator/start flow differs again).
  - Either adapt the client to the room protocol or use a matching older server version.
- `python train.py NUM_PLAYERS NUM_GAMES [TRAIN [N_GAMES [FROZEN_CHECKPOINT]]]`
  - `TRAIN` must be one of `1/true/yes` to enable training (explicit parsing; everything else, including `false`, means greedy eval).
  - `N_GAMES` (default 6) is the number of parallel games per update step, not the training length: each game gets its own server port starting at 6791 (`port += 1`), so 6 pooled games need server instances on ports 6791–6796. All `N_GAMES × NUM_PLAYERS` clients play concurrently under frozen weights and their rollouts are pooled into **one** PPO update. Training runs so far used 6 pooled games — that's why `training_data.txt` has ~1 row per 6 games. Scale the update batch via `N_GAMES`, not `NUM_GAMES`.
  - `FROZEN_CHECKPOINT` (optional) puts a frozen past-self on the last seat of every pooled game: it plays greedily from that checkpoint and contributes no rollouts.
- `NUM_GAMES` is the number of pooled update steps: the loop reuses the same player objects; each `play_game()` opens a fresh websocket connection, then all players' pending events are flushed (`train_agents()`) and `shared_agent.train()` runs once per step.

## Architecture / file map

- `train.py` — entry point. Creates one shared `ActorCriticAgent` and one `WizardEnv` (websocket client) per player, each in a thread.
- `actor_critic.py` — `ActorCriticAgent`: the **active** shared PPO policy. Shared trunk + value head, separate 21-action bid head and 60-action play head. Checkpoints in `training_actor_critic/shared_policy.weights.h5`.
- `environment.py` — `WizardEnv`: websocket protocol handling, fixed 327-dim state encoding, legal-action computation, reward shaping.
- `q_agent.py` — abandoned earlier PPO draft (kept for reference only; name survived the DQN era). Do not use, extend, or load its checkpoints.
- Checkpoints are incompatible across implementations (`actor_critic.py` vs `q_agent.py` vs old DQN) — do not mix them.
- Versioned snapshots like `training_actor_critic/shared_policy-v2-24000.weights.h5` are made by manually copying the checkpoint. Code only ever reads/writes the unversioned `shared_policy.weights.h5`.

## Conventions & gotchas

- `training_data.txt` (3 legacy columns: player loss, bid CE loss, critic loss — parseable by the Julia notebook, don't change its format), `training_metrics.txt` (health metrics per update: update, player loss, critic loss, approx KL, clip fraction, bid/play rewards), `game_data.txt` (bids/tricks per player-game) and `game_meta.txt` (one metadata line per `game_data.txt` line: timestamp, username, player count, phase1 flag, train flag, agent update counter) are append-only logs written by the code. Numbered variants (`-v2-36000`, `.old`, `-didntwork`) are manually archived logs. `notebooks/plot_losses.ipynb` and `notebooks/plot_progress.ipynb` plot these files.
- `ActorCriticAgent.phase1` switches reward semantics everywhere: `True` = bid head trained via cross-entropy against tricks won + greedy trick rewards; `False` (current) = PPO with score-delta bid rewards. Both `environment.py` (`on_player_update`) and `actor_critic.py` (`train`) branch on it — keep them consistent.
- Since the "Credit round score to the play stream" change, the round score delta is added to **both** the bid stream and the play stream's last pending action (terminal GAE reward that propagates back through the round's play decisions). `game_data.txt` rows logged before that commit are not comparable to rows after it; use `game_meta.txt` to find the boundary.
- Card actions are the server's card IDs (0–59 fixed deck slots), not hand indices. Bid actions are `0..round`; the bid that would make the announced total equal the round is removed for the last bidder.
- Trump choice is not learned: heuristic in `WizardEnv._choose_trump_color` (most cards ≥ 11, tie-break by suit length, fallback red when the hand is hidden in round 1).
- lr and entropy bonus anneal linearly over the first 10k updates (`_anneal_hyperparams`), but `self.epoch` is **not persisted** — the schedule restarts with every process invocation.
- `training_*/` and `__pycache__` are gitignored; checkpoints never go to git. `train()` overwrites `shared_policy.weights.h5` every update and also saves a versioned `-auto-NNNNNN` copy every 100 updates; older `-v2-NNNNN` files are manual copies.
- Active branch is `wip` (ahead of `master`); remote is `github.com/David96/wizard-ki`.

## Dependencies

- No requirements file exists. Imports: `tensorflow` (Keras 3 style, incl. `from keras import ops`), `websocket-client`, `numpy`. Pick a TensorFlow version compatible with your Python version.

## Reference docs

- `README.md` describes the `q_agent.py` shared-PPO rewrite design plus integration caveats (server state schema, reward/GAE boundaries, websocket lifecycle) that also apply to the active code.
- `README_ppo_integration.md` documents how `QAgent` replaced the old DQN agent and which assumptions to re-verify.
