# Lawnmower Path Optimizer 🚜

A physically accurate lawn mowing path optimization system with realistic blade coverage, multiple algorithms, and reinforcement learning.

## Overview

This project simulates and optimizes lawn mowing paths with realistic physics:
- **Grid squares = blade diameter / 2** (e.g., 70cm blade → 35cm squares)
- **Blade covers 2×2 grid cells** per position
- **Mower moves in 2-cell steps** (one blade diameter)
- **Realistic overlap modeling** between adjacent passes

Features six algorithms from classical pathfinding to modern deep reinforcement learning.

---

## Quick Start

```bash
cd lawnmower_optimizer
pip install -r requirements.txt

# Try different algorithms
python main.py maps/maze6.txt --algorithm greedy
python main.py maps/maze6.txt --algorithm mst
python main.py maps/maze6.txt --compare greedy astar mst
```

---

## Physical Model

### Grid System

**Blade coverage:**
```
Blade diameter = 70cm
Grid square = 35cm (blade_diameter / 2)

Mower at position (3,3):
+---+---+---+---+
| 2 | 3 | 4 | 5 |  
+---+---+---+---+
| ░ | ░ |   |   |  2  ← Blade covers
+---+---+---+---+      these 4 cells
| ░ | ● | ░ |   |  3
+---+---+---+---+
|   | ░ | ░ |   |  4
+---+---+---+---+

● = Mower center (odd coordinates)
░ = Covered by blade
```

### Movement

- Mower center at **odd coordinates only**: (1,1), (3,1), (5,3), etc.
- Each move advances **2 cells** (one blade diameter)
- 8 directions: N, NE, E, SE, S, SW, W, NW (all by 2 cells)

### Map Conversion

Original maps are automatically converted to 2× resolution:

**Input map:**
```
+++++++
+LLLLL+
+LLLLL+
+++++++
```

**Converted (2× resolution):**
```
++++++++++++++
+LLLLLLLLLL++
+LLLLLLLLLL++
+LLLLLLLLLL++
+LLLLLLLLLL++
++++++++++++++
```

Each original cell becomes a 2×2 block.

---

## Algorithms

### 1. Greedy (Look-Ahead)

Fast heuristic with directional bias:
- Looks ahead in all 8 directions
- Scores based on unmowed cells visible
- Prefers continuing straight (fewer turns)

```bash
python main.py maps/maze6.txt --algorithm greedy
```

**Performance:** ~70% coverage, fast execution

---

### 2. A* Coverage

Nearest-unmowed-cell strategy:
- Finds nearest unmowed cell
- Uses A* to navigate there
- Repeats until complete

```bash
python main.py maps/maze6.txt --algorithm astar
```

**Performance:** 100% coverage, ~34% efficiency

---

### 3. Boustrophedon (Back-and-Forth)

Classic systematic pattern:
- Sweeps in straight lines
- Reverses direction each row
- Deterministic (results cached)

```bash
python main.py maps/maze6.txt --algorithm boustrophedon
```

**Performance:** 100% coverage, ~38% efficiency

---

### 4. MST (Minimum Spanning Tree)

Graph-based optimization:
- Builds MST of all lawn positions
- Traverses tree depth-first
- Uses A* for jumps between branches

```bash
python main.py maps/maze6.txt --algorithm mst
```

**Performance:** 100% coverage, ~49% efficiency, cached

---

### 5. Hamiltonian Path

Optimal single-pass coverage:
- Visits all positions exactly once
- Backtracking with heuristics
- 30-second timeout with greedy fallback

```bash
python main.py maps/maze6.txt --algorithm hamiltonian
```

**Performance:** 100% efficiency when found, slow on large maps

---

### 6. DQN (Deep Q-Network)

Reinforcement learning agent:
- Learns from experience
- Neural network policy
- Auto-saves best model

**Train:**
```bash
python train_dqn.py maps/maze6.txt --train --episodes 1000
```

**Test:**
```bash
python train_dqn.py maps/maze6.txt --test
```

**Performance:** ~98% coverage, ~36% efficiency after training

---

## Algorithm Comparison

**maze6.txt (438 mower positions, 1752 total cells):**

| Algorithm | Coverage | Efficiency | Speed | Cached |
|-----------|----------|------------|-------|--------|
| Greedy | ~70% | ~24% | Fast | No |
| A* | 100% | ~34% | Medium | No |
| Boustrophedon | 100% | ~38% | Fast | Yes |
| MST | 100% | ~49% | Medium | Yes |
| Hamiltonian | 100% | 100% | Slow | Yes |
| DQN | ~98% | ~36% | Medium | No |

---

## Command Reference

### Main Script

```bash
python main.py <map_file> [options]

Options:
  -a, --algorithm ALGO      Run single algorithm
  -c, --compare ALGO...     Compare multiple algorithms
  --blade-diameter N        Blade diameter in cm (default: 70)
  --animate                 Create animation
  --save PATH               Save results
  --no-viz                  Disable visualization
  --no-cache                Disable path caching
  --clear-cache             Clear cache before running

Algorithms:
  greedy, astar, boustrophedon, mst, hamiltonian

Examples:
  python main.py maps/maze6.txt -a mst
  python main.py maps/maze6.txt -c greedy astar mst
  python main.py maps/maze6.txt -a astar --animate --save results/astar
  python main.py maps/maze6.txt -a greedy --blade-diameter 50
```

### RL Training

```bash
python train_dqn.py <map_file> [options]

Training:
  --train                   Train the agent
  --episodes N              Training episodes (default: 1000)
  --patience N              Early stopping patience (default: 100)
  --model PATH              Model save/load path
  --save-all                Save all episodes, not just best

Testing:
  --test                    Test trained agent
  --no-viz                  Disable visualization

Examples:
  python train_dqn.py maps/maze6.txt --train --episodes 1000
  python train_dqn.py maps/maze6.txt --test
```

---

## Creating Maps

### Text Grid Format

```
++++++++++++++
+LLLLLLLLLLLL+
+LLLL+++LLLLL+
+LLLL+++LLLLL+
+LLLLLLLLLLLL+
++++++++++++++

+ = Obstacle (wall, tree, building)
L = Lawn (to be mowed)
S = Start position (optional)
E = End position (optional)
```

### Generate Random Map

```bash
python create_map.py random \
    --size 30 30 \
    --density 0.2 \
    --output maps/random.txt
```

### Convert Image

```bash
python create_map.py image \
    --input yard.png \
    --size 50 50 \
    --threshold 128 \
    --output maps/yard.txt
```

---

## Physical Parameters

### Blade Diameter

Controls grid resolution and coverage:

```bash
# Standard 70cm blade → 35cm squares
python main.py maps/maze6.txt -a greedy

# Smaller 50cm blade → 25cm squares (finer grid)
python main.py maps/maze6.txt -a greedy --blade-diameter 50

# Larger 90cm blade → 45cm squares (coarser grid)
python main.py maps/maze6.txt -a greedy --blade-diameter 90
```

**Effects:**
- **Larger blade** = coarser grid, fewer positions, faster computation
- **Smaller blade** = finer grid, more positions, higher precision

---

## Diagonal Movement

Diagonal moves are validated to prevent cutting through obstacle corners:

```
Valid diagonal:          Invalid diagonal:
L  L                     +  L
 \/  ✓                    \/  ✗
 /\                       /\
M                        M

Both adjacent            One adjacent
cells free               cell blocked
```

The circular blade would hit obstacles in invalid cases.

---

## Advanced Features

### Path Caching

Deterministic algorithms (Boustrophedon, MST, Hamiltonian) cache results:

```bash
# First run - computes path
python main.py maps/maze6.txt -a mst
# ✓ Cached path (892 steps)

# Second run - instant!
python main.py maps/maze6.txt -a mst
# ✓ Loaded cached path (892 steps)

# Clear cache
python main.py maps/maze6.txt --clear-cache
```

### Auto-Save Best Model (RL)

DQN automatically saves when coverage improves:

```bash
python train_dqn.py maps/maze6.txt --train

# Episode 50 | Coverage: 89.2%
# Model saved ✓
#
# Episode 100 | Coverage: 95.1%  ← Better!
# Model saved ✓
```

### Early Stopping (RL)

Stops training when no improvement:

```bash
python train_dqn.py maps/maze6.txt --train --patience 100
# Stops after 100 episodes without improvement
```

### Learning Rate Decay (RL)

Exponential decay for better convergence:
```
Episode   1: LR = 0.00100
Episode 500: LR = 0.00065
Episode 1000: LR = 0.00045
```

---

## Performance Metrics

### Coverage (%)
Percentage of lawn cells mowed:
- **100%** = All lawn mowed (ideal)
- **<100%** = Some lawn missed

### Efficiency (%)
Percentage of moves that mowed new cells:
- **100%** = Perfect (every step mowed new cell)
- **30-40%** = Typical (includes navigation)

### Total Steps
Number of moves taken:
- Lower is better
- Minimum = number of mower positions

---

## Tips & Best Practices

### Choosing an Algorithm

**For guaranteed complete coverage:**
- **MST** - Best efficiency (~49%)
- **Boustrophedon** - Simple and fast
- **Hamiltonian** - Optimal but slow

**For speed:**
- **Greedy** - Fastest, partial coverage
- **Boustrophedon** - Fast with caching

**For learning-based:**
- **DQN** - Train once, reuse many times

### Blade Diameter Selection

**Larger blades (80-100cm):**
- Fewer positions to visit
- Faster computation
- Harder to navigate tight spaces

**Standard blades (60-70cm):**
- Good balance
- Recommended for most cases

**Smaller blades (40-50cm):**
- Better for complex layouts
- More precise control
- Takes longer to compute

---

## Project Structure

```
lawnmower_optimizer/
├── algorithms/              # All 6 algorithms
│   ├── greedy.py
│   ├── astar.py
│   ├── boustrophedon.py
│   ├── mst.py
│   ├── hamiltonian.py
│   └── dqn_agent.py
│
├── utils/
│   ├── environment.py      # Grid physics & blade coverage
│   ├── path_cache.py       # Caching system
│   └── config.py
│
├── visualization/
│   └── plotter.py          # Matplotlib visualization
│
├── maps/                    # Map files
├── cache/                   # Cached paths
├── models/                  # Trained RL models
│
├── main.py                  # Main script
├── train_dqn.py            # RL training
├── create_map.py           # Map generation
├── example.py              # Quick demo
│
└── README.md               # This file
```

---

## Documentation

- **README.md** - This file (overview & usage)
- **FINE_GRID_MODE.md** - Technical details of grid system
- **DIAGONAL_FIX.md** - Diagonal movement validation
- **ALL_ALGORITHM_FIXES.md** - Algorithm updates
- **QUICKSTART.md** - Quick start guide

---

## Troubleshooting

### "Invalid move detected"
The algorithm tried to cut through an obstacle corner. This has been fixed in all algorithms.

### Low coverage percentage
Some algorithms (like Greedy) may not reach 100% coverage on complex maps. Try MST or Boustrophedon for complete coverage.

### Hamiltonian timeout
Normal for large maps. Try MST instead for similar efficiency.

### Import errors
Ensure you're running from the project root:
```bash
cd lawnmower_optimizer
python main.py ...
```

---

## Example Session

```bash
$ python main.py maps/maze6.txt --algorithm mst

Loading map: maps/maze6.txt
Converted map: 31×30 → 62×60 (fine grid)
Mower blade diameter: 70.0cm
Grid square size: 35.0cm
Valid mower positions: 438
Total lawn cells: 1752

Running: MST
Building Minimum Spanning Tree...
MST built with 437 edges
MST completed: 1752/1752 cells
Total steps: 479
Efficiency: 91.4%

Results:
  Steps: 479
  Coverage: 100.0%
  Efficiency: 91.4%
```

---

## Future Enhancements

- [ ] Variable blade overlap patterns
- [ ] Path smoothing for realistic physics
- [ ] Battery/fuel constraints
- [ ] Multiple mowers (coordination)
- [ ] 3D terrain support
- [ ] Google Maps API integration

---

## License

MIT License - Free for any purpose

---

**Realistic lawn mowing simulation with physically accurate blade coverage!** 🚜🌱
