# PPO Fine-Tuning Documentation

## Overview

The system now includes **TWO training phases**:

1. **Phase 1: Imitation Learning** (Behavioral Cloning)
   - Fast, stable baseline
   - Learns from expert demonstrations
   - Achieves 95-100% coverage

2. **Phase 2: PPO Fine-tuning** ← NEW!
   - True Reinforcement Learning
   - Discovers improvements beyond experts
   - Optimizes for efficiency and smoothness

## What is PPO?

**PPO (Proximal Policy Optimization)** is a state-of-the-art RL algorithm that:
- Uses policy gradients to optimize actions
- Includes "clipping" to prevent destructive updates
- Is stable and sample-efficient
- Works well for continuous improvement

## Why Add PPO After Imitation Learning?

### Imitation Learning Limitations:
- ❌ Can only match experts (never exceed)
- ❌ Learns expert mistakes too
- ❌ No exploration of alternatives
- ❌ Limited to demonstrated situations

### PPO Advantages:
- ✅ Can discover novel strategies
- ✅ Learns from experience and rewards
- ✅ Explores alternative paths
- ✅ Optimizes for specific objectives (efficiency, smoothness)

### Combined Approach (Best of Both):
1. **IL gives warm start** → Fast learning, good baseline
2. **PPO refines** → Discovers optimizations

## Architecture

### Additional Components

**Value Network** (Critic):
```
Input (65) → Dense(256) → ReLU → Dropout
          → Dense(256) → ReLU → Dropout
          → Dense(128) → ReLU
          → Dense(1) → Value estimate
```

Estimates expected future reward from each state.

**PPO Algorithm**:
```
1. Collect trajectories with current policy
2. Compute advantages using GAE
3. Update policy with clipped objective:
   L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
4. Update value network
5. Repeat
```

### Reward Function

The agent is optimized for:

```python
reward = 0

# Main objective: Coverage
reward += new_cells_covered * 10.0

# Efficiency: Minimize steps
reward -= 0.1 per step

# Smoothness: Minimize direction changes
reward -= 0.5 per direction change

# Completion bonus
reward += 100.0 for 100% coverage
reward += bonus for quick completion

# Progress: Move toward uncovered areas
reward += 0.1 / (distance_to_uncovered + 1)
```

## Usage

### Quick Start

```bash
# Train with IL + PPO (recommended)
python quickstart.py train --ppo

# Quick mode (for testing, 100 iterations)
python quickstart.py train --ppo --quick
```

### Advanced Training

```bash
python train_with_ppo.py \
    --ppo-iterations 1000 \
    --ppo-trajectories 4 \
    --ppo-epochs 4 \
    --ppo-lr 3e-4 \
    --gamma 0.99 \
    --clip-epsilon 0.2
```

### Skip Phases

```bash
# Skip IL (use pretrained)
python train_with_ppo.py --skip-il

# IL only (no PPO)
python train_with_ppo.py --skip-ppo
```

## Hyperparameters

### PPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ppo_iterations` | 500 | Number of training iterations |
| `ppo_trajectories` | 4 | Trajectories per iteration |
| `ppo_epochs` | 4 | PPO updates per batch |
| `ppo_lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `value_loss_coef` | 0.5 | Value loss coefficient |
| `entropy_coef` | 0.01 | Entropy bonus |

### Tuning Guide

**For faster learning:**
- Increase `ppo_trajectories` (more data per update)
- Increase `ppo_lr` (bigger steps)
- Decrease `ppo_epochs` (less overfitting)

**For stability:**
- Decrease `ppo_lr` (smaller steps)
- Increase `ppo_epochs` (more thorough updates)
- Decrease `clip_epsilon` (more conservative)

**For exploration:**
- Increase `entropy_coef` (more random actions)
- Decrease `gamma` (prioritize immediate rewards)

## Expected Results

### Training Time

| Phase | Time (CPU) | Episodes | Iterations |
|-------|-----------|----------|-----------|
| **Phase 1: IL** | 2-5 min | N/A | 50 epochs |
| **Phase 2: PPO** | 20-40 min | ~2000 | 500 iterations |
| **Total** | 22-45 min | | |

With GPU: 5-10x faster

### Performance Improvement

Starting from IL baseline:

| Metric | After IL | After PPO | Improvement |
|--------|----------|-----------|-------------|
| **Coverage** | 95-100% | 97-100% | +0-5% |
| **Steps** | 1000-1200 | 950-1100 | -50 to -100 |
| **Efficiency** | 140-160% | 150-175% | +10-15% |
| **Direction Changes** | 200-400 | 150-300 | -50 to -100 |

PPO should:
- ✅ Improve efficiency (fewer steps for same coverage)
- ✅ Reduce direction changes (smoother paths)
- ✅ Maintain or improve coverage
- ✅ Discover shortcuts experts missed

## Training Curves

The system generates comprehensive visualizations:

### Phase 1 (Imitation Learning):
- Training loss (should decrease)
- Validation accuracy (should increase to 75-90%)

### Phase 2 (PPO):
- **Episode rewards** (should increase)
- **Episode lengths** (should decrease)
- **Policy loss** (fluctuates, that's normal)
- **Value loss** (should decrease)
- **Entropy** (should start high, gradually decrease)

### What to Look For:

**Good training:**
- 📈 Rewards trending upward
- 📉 Episode lengths decreasing
- 📉 Entropy gradually decreasing (0.5 → 0.1)

**Problems:**
- 📉 Rewards staying flat or decreasing → increase learning rate or entropy
- 📈 Episode lengths increasing → agent getting worse, reduce LR
- 🔄 High variance → increase trajectories per iteration

## Code Structure

```
ppo_trainer.py (560 lines)
├── ValueNetwork          - Critic for value estimation
├── PPOTrajectory        - Stores episode data
├── PPOTrainer           - Main PPO training loop
│   ├── compute_gae()    - Generalized Advantage Estimation
│   ├── compute_reward() - Reward function
│   ├── collect_trajectory() - Rollout policy
│   ├── update_policy()  - PPO update step
│   ├── train()          - Main training loop
│   └── evaluate()       - Test performance
```

## Comparison: IL vs IL+PPO

### Imitation Learning Alone

**Pros:**
- ✅ Fast (2-5 minutes)
- ✅ Stable (always converges)
- ✅ Good baseline (matches experts)
- ✅ Simple (no reward engineering)

**Cons:**
- ❌ Limited by expert quality
- ❌ Can't discover improvements
- ❌ May learn expert mistakes

### IL + PPO

**Pros:**
- ✅ Best of both worlds
- ✅ Warm start from IL
- ✅ Can exceed expert performance
- ✅ Optimized for specific objectives
- ✅ Discovers novel strategies

**Cons:**
- ❌ Slower (20-40 minutes total)
- ❌ More complex (more hyperparameters)
- ❌ Requires reward engineering

### Recommendation:

- **Use IL only** if:
  - You need fast results
  - Expert performance is sufficient
  - You want simplicity

- **Use IL + PPO** if:
  - You want best possible performance
  - You have time for training
  - You want to beat experts

## Troubleshooting

### Issue: Rewards not improving

**Check:**
1. Is coverage the bottleneck? → Adjust coverage reward weight
2. Is agent exploring? → Increase entropy coefficient
3. Is learning rate too low? → Increase to 5e-4 or 1e-3

**Solution:**
```bash
python train_with_ppo.py --ppo-lr 5e-4 --entropy-coef 0.02
```

### Issue: Agent gets worse during PPO

**Cause:** Learning rate too high, destroying IL knowledge

**Solution:** Reduce learning rate
```bash
python train_with_ppo.py --ppo-lr 1e-4 --clip-epsilon 0.1
```

### Issue: Training too slow

**Solutions:**
1. Reduce iterations: `--ppo-iterations 200`
2. Use quick mode: `--quick`
3. Use GPU: `--device cuda`
4. Reduce trajectories per iteration: `--ppo-trajectories 2`

### Issue: High variance in results

**Solutions:**
1. Increase trajectories: `--ppo-trajectories 8`
2. Increase PPO epochs: `--ppo-epochs 6`
3. Reduce learning rate: `--ppo-lr 1e-4`

## Advanced: Custom Reward Functions

You can modify the reward function in `ppo_trainer.py`:

```python
def compute_reward(self, state_info: dict) -> float:
    reward = 0.0
    
    # Customize these weights
    reward += new_coverage * 10.0        # Coverage weight
    reward -= 0.1                        # Step penalty
    reward -= direction_change * 0.5     # Turn penalty
    reward += completion_bonus * 100     # Completion bonus
    
    # Add your own rewards
    reward += custom_metric * weight
    
    return reward
```

**Examples:**

Prioritize speed:
```python
reward -= 0.5  # Larger step penalty
```

Prioritize coverage:
```python
reward += new_coverage * 20.0  # Double coverage reward
```

Minimize turns:
```python
reward -= direction_change * 2.0  # Bigger turn penalty
```

## Evaluation

After training, the agent is evaluated on:

1. **Coverage %** - Proportion of area mowed
2. **Steps** - Total number of moves
3. **Efficiency %** - Coverage / Steps ratio
4. **Direction Changes** - Number of turns

Results are saved to:
- `models/best_ppo_model.pt` - Best model during training
- `models/final_ppo_model.pt` - Final model
- `plots/complete_training_curves.png` - All training curves
- Console output with detailed metrics

## Files Created

```
New files:
├── ppo_trainer.py           - PPO implementation (560 lines)
├── train_with_ppo.py        - Combined training script (350 lines)
└── PPO_DOCUMENTATION.md     - This file

Updated files:
├── quickstart.py            - Added --ppo flag
└── (train_rl_agent.py unchanged, still works standalone)
```

## Next Steps

1. **Train your agent:**
   ```bash
   python quickstart.py train --ppo
   ```

2. **Monitor training:**
   - Watch console output for rewards
   - Check `plots/complete_training_curves.png`

3. **Evaluate:**
   ```bash
   python quickstart.py test
   ```

4. **Experiment:**
   - Adjust hyperparameters
   - Modify reward function
   - Train on different maps

## Summary

You now have a **complete RL system** with:

✅ **Phase 1: Imitation Learning** (fast baseline)  
✅ **Phase 2: PPO Fine-tuning** (true RL optimization)  
✅ **Flexible training** (IL-only or IL+PPO)  
✅ **Comprehensive evaluation**  
✅ **Full visualization**  
✅ **Well-documented**  

The agent can now **learn from experts AND discover improvements beyond them**! 🎉
