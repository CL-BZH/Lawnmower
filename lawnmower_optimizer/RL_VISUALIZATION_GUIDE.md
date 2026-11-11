# How to Visualize the RL Agent in main.py

## ✅ What Was Added

The RL agent is now integrated into `main.py` and can be visualized just like other algorithms!

---

## 🚀 Usage Examples

### 1. **Run RL Agent with Visualization**

```bash
python main.py maps/maze6.txt --algorithm rl_agent
```

This will:
- Load the trained RL model (`models/best_ppo_model.pt`)
- Run the agent on the map
- Show visualization of the path
- Print coverage and efficiency metrics

### 2. **Animate the RL Agent**

```bash
python main.py maps/maze6.txt --algorithm rl_agent --animate
```

Creates an animated GIF showing the RL agent mowing step-by-step!

### 3. **Save Visualization**

```bash
python main.py maps/maze6.txt --algorithm rl_agent --save results/rl_test
```

Saves the visualization to `results/rl_test_rl_agent.png`

### 4. **Use a Different Model**

```bash
# Use IL model instead of PPO
python main.py maps/maze6.txt --algorithm rl_agent --rl-model models/best_pretrained_model.pt

# Use final PPO model
python main.py maps/maze6.txt --algorithm rl_agent --rl-model models/final_ppo_model.pt
```

### 5. **Compare RL Agent with Other Algorithms**

```bash
python main.py maps/maze6.txt --compare hybrid ccpp sweepopt rl_agent
```

This will:
- Run all 4 algorithms
- Print comparison table
- Show which is best

Example output:
```
================================================================================
Algorithm            Steps      Coverage     Efficiency   Time       Unmowed   
================================================================================
hybrid               1076       95.4         153.5        0.234      79        
ccpp                 1365       98.8         125.3        0.456      21        
sweepopt             1105       97.0         151.9        0.189      52        
rl_agent             1523       98.6         112.0        2.345      24        
================================================================================

🏆 BEST ALGORITHM: CCPP
   Coverage: 98.8%
   Steps: 1365
   Efficiency: 125.3%
   Unmowed cells: 21
```

### 6. **Animate and Save Multiple Algorithms**

```bash
python main.py maps/maze6.txt --compare hybrid ccpp rl_agent --animate --save results/comparison
```

Creates animated GIFs for all algorithms:
- `results/comparison_hybrid.gif`
- `results/comparison_ccpp.gif`
- `results/comparison_rl_agent.gif`

---

## 📊 Understanding the Output

When you run the RL agent, you'll see:

```
============================================================
Running: RL_AGENT
============================================================
  Loading RL agent...
  Loaded model from models/best_ppo_model.pt
  Running RL agent...
  Executing algorithm...
  Calculating coverage...

Results:
  Total steps: 1523
  Coverage: 98.6% (1707/1731)
  Efficiency: 112.0%
  Time: 2.345s
```

**Metrics explained:**
- **Steps**: Number of moves the agent made
- **Coverage**: Percentage of lawn cells mowed
- **Efficiency**: (Cells mowed / Steps) × 100
- **Time**: How long it took to run

---

## 🎨 Visualization Features

The visualization shows:
- **Gray**: Obstacles
- **White**: Unmowed lawn
- **Green**: Mowed lawn
- **Blue line**: Path taken by agent
- **Red dot**: Starting position
- **Green dot**: Current/ending position

---

## ⚙️ Advanced Options

### Control Maximum Steps

The RL agent will run for up to 2500 steps by default. To change:

```python
# Edit main.py, line ~143, change:
for step in range(kwargs.get('max_steps', 2500)):
# To higher value like:
for step in range(kwargs.get('max_steps', 3000)):
```

### Use Stochastic Policy

By default, the agent uses deterministic policy (always chooses best action). To use stochastic:

```python
# Edit main.py, line ~149, change:
action = agent.policy.get_action(state_tensor, deterministic=True).item()
# To:
action = agent.policy.get_action(state_tensor, deterministic=False).item()
```

This will make the agent sample actions probabilistically (more exploration).

---

## 🐛 Troubleshooting

### Error: "No trained RL model found"

**Solution**: Train the model first:
```bash
python train_with_ppo.py
```

### Error: "cannot import name 'LawnmowerRLAgent'"

**Solution**: Make sure you're running from the correct directory:
```bash
cd /path/to/lawnmower_optimizer
python main.py maps/maze6.txt --algorithm rl_agent
```

### RL Agent Takes Too Long

The RL agent is slower than algorithmic methods because it:
1. Loads neural network
2. Encodes state at each step (308 features!)
3. Runs forward pass through network

**Solution**: This is normal. The agent should complete in 2-10 seconds.

### Coverage Lower Than Expected

If you see coverage like 90% instead of 98%:
- Make sure you're loading the correct model
- Check which model was loaded (PPO vs IL)
- Try: `--rl-model models/best_ppo_model.pt`

---

## 📋 Quick Reference

| Command | What It Does |
|---------|--------------|
| `--algorithm rl_agent` | Run RL agent with visualization |
| `--animate` | Create animated GIF |
| `--save PATH` | Save visualization to file |
| `--rl-model PATH` | Use specific RL model |
| `--compare ... rl_agent` | Compare with other algorithms |
| `--no-viz` | Run without visualization |

---

## 🎯 Recommended Workflow

1. **Train the agent**:
   ```bash
   python train_with_ppo.py
   ```

2. **Test it standalone**:
   ```bash
   python test_rl_agent.py
   ```

3. **Visualize it**:
   ```bash
   python main.py maps/maze6.txt --algorithm rl_agent
   ```

4. **Compare with experts**:
   ```bash
   python main.py maps/maze6.txt --compare hybrid ccpp sweepopt rl_agent
   ```

5. **Create animations for presentation**:
   ```bash
   python main.py maps/maze6.txt --compare hybrid ccpp rl_agent --animate --save presentation/demo
   ```

---

## 🎉 You're All Set!

The RL agent is now fully integrated and can be visualized like any other algorithm!

Try it:
```bash
python main.py maps/maze6.txt --algorithm rl_agent
```
