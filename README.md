# DDA in 2D Puzzle Games using Reinforcement Learning

A CPU-friendly research prototype that studies **Dynamic Difficulty Adjustment (DDA)** in 2D key–door maze puzzles using Reinforcement Learning. An RL "director" agent adaptively adjusts puzzle difficulty parameters to keep player solve times within a target engagement window.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RL Director Agent                     │
│               (PPO / DQN via Stable-Baselines3)          │
│                                                         │
│  Observes: player performance, progress, difficulty      │
│  Actions:  adjust grid_size, keys, enemies, hints        │
│  Reward:   +1 in-window, −0.5 too-easy, −1 too-hard     │
└────────────────────────┬────────────────────────────────┘
                         │ difficulty params
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Procedural Level Generator                   │
│         (DFS maze + key-door placement + BFS check)       │
└────────────────────────┬────────────────────────────────┘
                         │ generated level
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  2D Puzzle Game Engine                    │
│          (grid, tiles, movement, key/door logic)          │
└────────────────────────┬────────────────────────────────┘
                         │ game state
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Bot Players (simulated)                      │
│     RandomBot │ BFSBot │ OptimalBot │ NoisyBFSBot         │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
rl_dda_puzzle/
├── envs/                  # Game engine & Gymnasium env
│   ├── puzzle_game.py     # Core game: grid, tiles, movement, level generation
│   └── dda_env.py         # DDAEnv: Gymnasium wrapper for RL director
├── players/               # Simulated player bots
│   └── bot_players.py     # RandomBot, BFSBot, OptimalBot, NoisyBFSBot
├── agents/                # RL training & evaluation
│   ├── train_ppo.py       # PPO training with SB3
│   ├── train_dqn.py       # DQN training with SB3
│   └── eval_agent.py      # Evaluation across all modes
├── configs/               # Hyperparameter configs
│   ├── ppo_config.yaml
│   └── dqn_config.yaml
├── notebooks/             # Analysis & visualization
│   └── analysis.ipynb
├── scripts/               # Automation scripts
│   └── run_bot_matches.ps1
├── results/               # Output artifacts
│   ├── logs/              # Training logs (CSV / W&B)
│   └── models/            # Saved model checkpoints
├── tests/                 # Unit & integration tests
│   ├── test_puzzle_game.py
│   └── test_dda_env.py
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/Yashwanthkumar/2d_puzzlegame_rl.git
cd 2d_puzzlegame_rl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Train RL Director (PPO)
```bash
python agents/train_ppo.py --config configs/ppo_config.yaml --timesteps 100000
```

### 4. Train RL Director (DQN)
```bash
python agents/train_dqn.py --config configs/dqn_config.yaml --timesteps 100000
```

### 5. Evaluate
```bash
# Evaluate all modes: fixed, heuristic, PPO, DQN
python agents/eval_agent.py --mode all --episodes 100
```

### 6. Analyze Results
Open `notebooks/analysis.ipynb` in Jupyter for plots and ablation analysis.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Solve-time unit | Game steps (not wall-clock) | Deterministic, reproducible |
| RL algorithms | PPO (primary) + DQN (baseline) | PPO is stable with small MLPs; DQN provides a comparison |
| Policy size | MLP 64×64 | CPU-friendly, fast training |
| Key colors | 3 (R, G, B) | Enough variety without combinatorial explosion |
| Bot players | NoisyBFSBot(noise_prob) | Smoothly simulates skill spectrum from random to optimal |
| Target window | 50–200 steps | Calibrated for 12×12 grids with 1–2 keys |

## Metrics

- **In-window rate**: Fraction of levels solved within target time window
- **Mean solve time**: Average steps to solve (with std dev)
- **Failure rate**: Fraction of timeouts / deaths
- **Session length**: Levels completed per session

## References

- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- Dynamic Difficulty Adjustment via RL: Andrade et al. (2006), Zook & Riedl (2012)
- PCG via RL: Khalifa et al. (2020)

## License

This project is for research and educational purposes.
