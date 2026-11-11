# Quick Start Guide - Lawnmower Optimizer

## Installation (3 minutes)

```bash
cd lawnmower_optimizer
pip install -r requirements.txt
```

## Try It Now! (5 examples)

### 1. Compare All Algorithms (Recommended first try)
```bash
python main.py maps/maze6.txt --compare greedy astar boustrophedon
```
**What happens:** Runs 3 algorithms and shows side-by-side comparison
**Output:** Performance metrics + visualization window

---

### 2. Watch an Animation
```bash
python main.py maps/maze6.txt --algorithm astar --animate
```
**What happens:** Animated mowing sequence
**Output:** Watch the mower move step-by-step!

---

### 3. Create Your Own Map
```bash
# Create a random 30x30 lawn
python create_map.py random --size 30 30 --density 0.2 --output maps/my_lawn.txt

# Test it
python main.py maps/my_lawn.txt --algorithm greedy
```
**What happens:** Generates random map, then mows it
**Output:** New map file + mowing visualization

---

### 4. Train AI Agent (Advanced)
```bash
# Train for 500 episodes (takes ~5 minutes)
python train_dqn.py maps/maze6.txt --train --episodes 500

# Test the trained agent
python train_dqn.py maps/maze6.txt --test
```
**What happens:** AI learns to mow through trial & error
**Output:** Trained model + test performance

---

### 5. Run Simple Example
```bash
python example.py
```
**What happens:** Runs all algorithms on example map
**Output:** Comparison of all methods

---

## Understanding the Output

When you run an algorithm, you'll see:

```
Running: GREEDY
============================================================
Greedy completed: 419/438 cells
Efficiency: 31.9%
Direction changes: 139

Results:
  Total steps: 1315
  Coverage: 95.7%
  Efficiency: 31.9%
```

**What this means:**
- **Coverage**: 95.7% of lawn was mowed (ideally 100%)
- **Efficiency**: 31.9% of steps mowed new cells (rest were navigation)
- **Steps**: 1315 total moves to complete
- **Direction changes**: 139 turns (fewer is smoother)

---

## Common Use Cases

### "I want to see which algorithm works best"
```bash
python main.py maps/maze6.txt --compare greedy astar boustrophedon
```

### "I want to create a map from a photo"
```bash
python create_map.py image --input my_yard.jpg --size 50 50 --output maps/my_yard.txt
python main.py maps/my_yard.txt --algorithm boustrophedon
```

### "I want to save the results"
```bash
python main.py maps/maze6.txt --algorithm greedy --save results/run1
# Creates: results/run1_greedy.png
```

### "I don't want the visualization (headless server)"
```bash
python main.py maps/maze6.txt --algorithm greedy --no-viz
```

---

## Project Structure

```
lawnmower_optimizer/
├── main.py              ← Main program (start here!)
├── example.py           ← Simple demo
├── create_map.py        ← Make custom maps
├── train_dqn.py         ← Train AI agent
│
├── algorithms/          ← All mowing algorithms
│   ├── greedy.py
│   ├── astar.py
│   ├── boustrophedon.py
│   ├── hamiltonian.py
│   └── dqn_agent.py
│
├── maps/                ← Grid map files
│   └── maze6.txt        ← Example map
│
└── README.md            ← Full documentation
```

---

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Try different maps** - create your own!
3. **Experiment with algorithms** - which is best for your map?
4. **Train the AI** - can it beat the classical algorithms?
5. **Contribute** - add new algorithms or features!

---

## Troubleshooting

**Problem:** Visualization doesn't show
**Solution:** Make sure matplotlib is installed: `pip install matplotlib`

**Problem:** "No module named 'utils'"
**Solution:** Run from project root: `cd lawnmower_optimizer && python main.py ...`

**Problem:** DQN training is slow
**Solution:** Reduce episodes: `--episodes 200` or use smaller map

---

## Need Help?

- Full docs: `README.md`
- Check example: `python example.py`
- Test installation: `python -c "from utils.environment import LawnmowerEnvironment; print('OK')"`

**Happy Mowing!** 🚜🌱
