"""
Configuration for lawnmower visualization and simulation
"""

# Grid display symbols
SYMBOLS = {
    'START': 'S',
    'END': 'E',
    'LAWN': 'L',
    'OBSTACLE': '+',
    'MOWED': '.',
    'UNMOWED': ' ',
}

# Colors for visualization
COLORS = {
    'mower': '#1f77b4',      # Blue
    'start': '#9467bd',      # Purple
    'end': '#ff7f0e',        # Orange
    'obstacle': '#2f2f2f',   # Dark gray
    'lawn': "#0a6b0a",       # Green
    'mowed': "#8ff798",      # Light green
    'unmowed': '#0a6b0a',    # Green
    'path': '#e377c2',       # Pink
    'grid': '#ffffff',       # White
}

# Visualization settings
VIZ_SETTINGS = {
    'figsize': (12, 10),
    'dpi': 100,
    'animation_interval': 50,  # milliseconds
    'save_animation': False,
}

# Algorithm parameters
ALGO_PARAMS = {
    'greedy': {
        'look_ahead_distance': 5,
    },
    'a_star': {
        'heuristic': 'euclidean',
    },
    'dqn': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'memory_size': 10000,
        'target_update': 10,
    },
}
