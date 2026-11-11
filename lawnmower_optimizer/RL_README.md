# Lawnmower Reinforcement Learning System

This system uses **Imitation Learning** to train an RL agent that learns from expert algorithms (hybrid, ccpp, sweepopt) to find optimal lawnmower paths.

## Overview

The RL agent learns to:
1. **Maximize coverage** (100% of mowable cells)
2. **Minimize steps** (efficient path)
3. **Minimize direction changes** (smooth movement)

## Architecture

### State Representation (65 features)
- **Current position** (2): Normalized (x, y)
- **Local occupancy grid** (25): 5x5 grid around robot (obstacle/lawn)
- **Local coverage grid** (25): 5x5 grid of mowed/unmowed status
- **Coverage statistics** (3): Coverage ratio, mowed count, remaining
- **Last action** (8): One-hot encoding of previous direction
- **Direction to nearest unmowed** (2): Vector to closest unmowed cell

### Action Space (8 discrete actions)
- North, Northeast, East, Southeast, South, Southwest, West, Northwest

### Network Architecture
- Input: 65 features
- Hidden: 256 → 256 → 128 neurons with ReLU and Dropout
- Output: 8 action probabilities (softmax)

### Training Method: Imitation Learning
1. **Collect demonstrations** from expert algorithms
2. **Supervised learning** to mimic expert actions
3. Policy network learns action distribution given state

## Installation

```bash
# Install PyTorch
pip install torch torchvision

# Other dependencies (should already be installed)
pip install numpy matplotlib
```

## Usage

### 1. Train the Agent

```bash
cd /mnt/user-data/outputs/lawnmower_optimizer
python train_rl_agent.py
```

This will:
- Collect demonstrations from hybrid, ccpp, and sweepopt algorithms
- Pre-train the policy network (50 epochs)
- Save the trained model to `models/best_pretrained_model.pt`
- Generate training curves in `plots/training_curves.png`

**Training parameters (in `train_rl_agent.py`):**
```python
# Demonstration collection
algorithms = ['hybrid', 'ccpp', 'sweepopt']
num_demos_per_algo = 3  # 3 paths per algorithm = 9 total demos

# Training
batch_size = 64
num_epochs = 50
learning_rate = 1e-3
validation_split = 0.2
```

### 2. Test the Agent

```bash
python test_rl_agent.py
```

This will:
- Load the trained agent
- Run evaluation episodes
- Compare against expert algorithms
- Generate visualizations in `plots/`

### 3. Use the Trained Agent in Your Code

```python
from train_rl_agent import LawnmowerRLAgent
from utils.environment import LawnmowerEnvironment
import torch

# Load environment and agent
env = LawnmowerEnvironment('maps/maze6.txt')
agent = LawnmowerRLAgent(env)
agent.load_model('models/best_pretrained_model.pt')

# Run agent to generate path
current_pos = env.start_pos
mowed_cells = set()
path = [current_pos]
last_action = None

for step in range(2000):
    # Update mowed cells
    blade = env.get_blade_coverage(current_pos)
    for cell in blade:
        if cell in env.reachable_lawn_cells:
            mowed_cells.add(cell)
    
    # Check if done
    if len(mowed_cells & env.reachable_lawn_cells) >= len(env.reachable_lawn_cells):
        break
    
    # Get action
    state = agent.encoder.encode_state(current_pos, mowed_cells, last_action)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        action = agent.policy.get_action(state_tensor, deterministic=True).item()
    
    # Execute action
    from train_rl_agent import ActionMapper
    dx, dy = ActionMapper.action_to_delta(action)
    next_pos = (current_pos[0] + dx, current_pos[1] + dy)
    
    if next_pos in env.lawn_cells:
        path.append(next_pos)
        current_pos = next_pos
        last_action = action

print(f"Path length: {len(path)}")
print(f"Coverage: {100 * len(mowed_cells) / len(env.reachable_lawn_cells):.1f}%")
```

## File Structure

```
lawnmower_optimizer/
├── train_rl_agent.py          # Main training script
├── test_rl_agent.py            # Testing and comparison script
├── models/                     # Saved models
│   ├── best_pretrained_model.pt
│   └── final_model.pt
├── data/                       # Demonstration data
│   ├── expert_demonstrations.pkl
│   └── evaluation_results.pkl
└── plots/                      # Visualizations
    ├── training_curves.png
    ├── algorithm_comparison.png
    ├── rl_agent_path.png
    ├── hybrid_path.png
    ├── ccpp_path.png
    └── sweepopt_path.png
```

## Key Classes

### `LawnmowerStateEncoder`
Converts game state into neural network features

### `PolicyNetwork`
Neural network that predicts action probabilities

### `LawnmowerRLAgent`
Main agent class with:
- `collect_demonstrations()`: Gather expert data
- `pretrain_from_demonstrations()`: Supervised learning
- `evaluate()`: Test performance
- `save_model()` / `load_model()`: Checkpoint management

### `ActionMapper`
Converts between discrete actions and grid movements

## Expected Results

After training, the RL agent should achieve:
- **Coverage: 95-100%** (learns to cover entire map)
- **Steps: 1000-1200** (competitive with experts)
- **Efficiency: 140-160%** (good coverage per step)
- **Direction Changes: 200-400** (smooth movement)

The agent should match or improve upon the expert algorithms by learning to combine their strengths.

## Advanced: Fine-tuning with RL

After imitation learning, you can optionally fine-tune with reinforcement learning:

```python
# Add to train_rl_agent.py after pre-training

def finetune_with_rl(agent, num_episodes=1000):
    """Fine-tune with PPO or other RL algorithm"""
    # This is a placeholder for future enhancement
    # Would use actual RL (PPO, SAC, etc.) to optimize beyond expert demonstrations
    pass
```

This would allow the agent to discover novel strategies beyond what the expert algorithms demonstrate.

## Troubleshooting

**Issue: Low training accuracy (<70%)**
- Collect more demonstrations: increase `num_demos_per_algo`
- Train longer: increase `num_epochs`
- Check that expert algorithms are working correctly

**Issue: Agent gets stuck during evaluation**
- The agent may need more diverse demonstrations
- Add epsilon-greedy exploration during evaluation
- Check for invalid move handling

**Issue: Poor coverage (<90%)**
- Expert algorithms may not provide good coverage examples
- Ensure demonstrations include complete coverage paths
- Add reward shaping for uncovered cells

## Next Steps

1. **Multi-map training**: Train on multiple maps for generalization
2. **Curriculum learning**: Start with small regions, gradually increase complexity
3. **RL fine-tuning**: Use PPO/SAC to optimize beyond imitation
4. **Reward shaping**: Add intermediate rewards for better learning
5. **Hyperparameter tuning**: Optimize network size, learning rate, etc.

## Citation

This implementation is based on:
- **Imitation Learning**: Learning from expert demonstrations
- **Behavioral Cloning**: Supervised learning of expert policies
- **Deep Learning**: Neural networks for policy representation

For questions or improvements, refer to the RL and path planning literature.
