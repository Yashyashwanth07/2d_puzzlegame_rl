# DDA Puzzle RL - Automated Bot Playtests
# ========================================
# Run this script to train and evaluate all DDA modes.

Write-Host "============================================"
Write-Host "  DDA Puzzle RL - Automated Evaluation"
Write-Host "============================================"
Write-Host ""

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Step 1: Run tests
Write-Host "[1/5] Running tests..."
Set-Location $ProjectRoot
python -m pytest tests/ -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed! Fix issues before proceeding." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Train PPO (short run for CI/demo)
Write-Host "[2/5] Training PPO (10k steps)..."
python agents/train_ppo.py --config configs/ppo_config.yaml --timesteps 10000 --no-wandb
Write-Host ""

# Step 3: Train DQN (short run for CI/demo)
Write-Host "[3/5] Training DQN (10k steps)..."
python agents/train_dqn.py --config configs/dqn_config.yaml --timesteps 10000 --no-wandb
Write-Host ""

# Step 4: Evaluate baselines (fixed + heuristic)
Write-Host "[4/5] Evaluating baselines..."
python agents/eval_agent.py --mode all --episodes 50 --seed 42
Write-Host ""

# Step 5: Summary
Write-Host "[5/5] Done! Check results/ for logs and models."
Write-Host ""
Write-Host "============================================"
Write-Host "  Evaluation complete!"
Write-Host "============================================"
