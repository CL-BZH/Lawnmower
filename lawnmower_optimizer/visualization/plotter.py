"""
Visualization module for lawn mowing paths
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
from typing import List, Tuple
from utils.environment import LawnmowerEnvironment, MowerState
from utils.config import COLORS


class LawnmowerVisualizer:
    """Visualize lawn mowing paths and algorithms"""
    
    def __init__(self, env: LawnmowerEnvironment, figsize=(12, 10)):
        """
        Initialize visualizer
        
        Args:
            env: LawnmowerEnvironment instance
            figsize: Figure size (width, height)
        """
        self.env = env
        self.figsize = figsize
        
    def plot_path(self, path: List[Tuple[int, int]], 
                  title: str = "Lawnmower Path",
                  save_file: str = None,
                  show_grid: bool = True):
        """
        Plot a complete path on the grid
        
        Args:
            path: List of (x, y) positions
            title: Plot title
            save_file: If provided, save to this file
            show_grid: Whether to show grid lines
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate mowed cells from path
        mowed_cells = set()
        for pos in path:
            coverage = self.env.get_blade_coverage(pos)
            for cell in coverage:
                if cell in self.env.all_lawn_cells:
                    mowed_cells.add(cell)
        
        # Draw grid with mowed cells
        self._draw_grid(ax, mowed_cells)
        
        # Draw path
        if len(path) > 1:
            xs = [x for x, y in path]
            ys = [y for x, y in path]
            ax.plot(xs, ys, color=COLORS['path'], linewidth=2, 
                   alpha=0.8, zorder=3, label='Path')
            
            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=COLORS['start'], 
                   markersize=15, zorder=4, label='Start')
            ax.plot(xs[-1], ys[-1], 's', color=COLORS['end'], 
                   markersize=15, zorder=4, label='End')
        
        # Calculate statistics
        mowed = set(path)
        total_lawn = len(self.env.lawn_cells)
        efficiency = 100 * len(mowed & self.env.lawn_cells) / len(path) if path else 0
        coverage = 100 * len(mowed & self.env.lawn_cells) / total_lawn if total_lawn > 0 else 0
        
        stats_text = (f"Steps: {len(path)}\n"
                     f"Coverage: {coverage:.1f}%\n"
                     f"Efficiency: {efficiency:.1f}%")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_file}")
        
        plt.show()
    
    def animate_path(self, path: List[Tuple[int, int]],
                    title: str = "Lawnmower Animation",
                    interval: int = 500,
                    save_file: str = None):
        """
        Create animated visualization of mowing path
        
        Args:
            path: List of (x, y) positions
            title: Animation title
            interval: Milliseconds between frames
            save_file: If provided, save animation to this file
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw initial grid (empty, we'll update cells as they're mowed)
        # Track cell patches for updating colors
        cell_patches = {}
        for y in range(self.env.height):
            for x in range(self.env.width):
                pos = (x, y)
                
                if pos in self.env.obstacle_cells:
                    color = COLORS['obstacle']
                elif pos in self.env.all_lawn_cells:
                    color = COLORS['unmowed']  # Start as unmowed
                else:
                    color = 'white'
                
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                        facecolor=color, 
                                        edgecolor='lightgray',
                                        linewidth=0.5,
                                        zorder=0)
                ax.add_patch(rect)
                cell_patches[(x, y)] = rect
        
        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.invert_yaxis()
        
        # Initialize path line and mower position
        line, = ax.plot([], [], color=COLORS['mowed'], linewidth=2, alpha=0.6)
        # Mower shown as large circle (diameter covers ~2 cells)
        mower, = ax.plot([], [], 'o', color=COLORS['mower'], markersize=30, alpha=0.5)
        
        # Stats text
        stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Track mowed cells for visualization
        mowed_cells = set()
        
        def init():
            line.set_data([], [])
            mower.set_data([], [])
            stats_text.set_text('')
            return line, mower, stats_text
        
        def animate(frame):
            # Update path
            xs = [x for x, y in path[:frame+1]]
            ys = [y for x, y in path[:frame+1]]
            line.set_data(xs, ys)
            
            # Update mower position (draw as larger circle representing blade)
            if frame < len(path):
                mower.set_data([path[frame][0]], [path[frame][1]])
                
                # Update mowed visualization - show blade coverage
                pos = path[frame]
                coverage = self.env.get_blade_coverage(pos)
                
                for cell in coverage:
                    if cell in self.env.all_lawn_cells and cell not in mowed_cells:
                        mowed_cells.add(cell)
                        # Change cell color from unmowed to mowed
                        if cell in cell_patches:
                            cell_patches[cell].set_facecolor(COLORS['mowed'])
            
            # Update stats
            total_mowed = len(mowed_cells)
            total_lawn = len(self.env.all_lawn_cells)
            efficiency = 100 * total_mowed / max(frame + 1, 1)
            coverage = 100 * total_mowed / total_lawn if total_lawn > 0 else 0
            stats = (f"Step: {frame + 1}/{len(path)}\n"
                    f"Mowed: {total_mowed}/{total_lawn}\n"
                    f"Coverage: {coverage:.1f}%\n"
                    f"Efficiency: {efficiency:.1f}%")
            stats_text.set_text(stats)
            
            return line, mower, stats_text
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(path), interval=interval,
                                      blit=False, repeat=True)
        
        if save_file:
            if save_file.endswith('.gif'):
                anim.save(save_file, writer='pillow', fps=1000/interval)
                print(f"Animation saved to {save_file}")
            elif save_file.endswith('.mp4'):
                anim.save(save_file, writer='ffmpeg', fps=1000/interval)
                print(f"Animation saved to {save_file}")
        
        plt.tight_layout()
        plt.show()
    
    def compare_algorithms(self, results: dict, save_file: str = None):
        """
        Compare multiple algorithm results
        
        Args:
            results: Dict of {algorithm_name: (path, metrics)}
            save_file: If provided, save comparison to this file
        """
        n_algos = len(results)
        fig, axes = plt.subplots(1, n_algos, figsize=(6*n_algos, 6))
        
        if n_algos == 1:
            axes = [axes]
        
        for ax, (name, (path, metrics)) in zip(axes, results.items()):
            # Draw grid
            self._draw_grid(ax)
            
            # Draw path
            if len(path) > 1:
                xs = [x for x, y in path]
                ys = [y for x, y in path]
                ax.plot(xs, ys, color=COLORS['path'], linewidth=1.5, alpha=0.6)
                ax.plot(xs[0], ys[0], 'o', color=COLORS['start'], markersize=10)
                ax.plot(xs[-1], ys[-1], 's', color=COLORS['end'], markersize=10)
            
            # Add metrics text
            stats = f"Steps: {metrics.get('steps', len(path))}\n"
            stats += f"Efficiency: {metrics.get('efficiency', 0):.1f}%\n"
            stats += f"Coverage: {metrics.get('coverage', 0):.1f}%"
            
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {save_file}")
        
        plt.show()
    
    def _draw_grid(self, ax, mowed_cells=None):
        """Draw the base grid with obstacles and lawn"""
        if mowed_cells is None:
            mowed_cells = set()
            
        # Create color map for grid
        for y in range(self.env.height):
            for x in range(self.env.width):
                pos = (x, y)
                
                if pos in self.env.obstacle_cells:
                    color = COLORS['obstacle']
                elif pos in mowed_cells:
                    # Mowed cells
                    color = COLORS['mowed']
                elif pos in self.env.all_lawn_cells:
                    # Unmowed lawn
                    color = COLORS['lawn']
                else:
                    color = 'white'
                
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                        facecolor=color, 
                                        edgecolor='lightgray',
                                        linewidth=0.5)
                ax.add_patch(rect)
        
        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.invert_yaxis()  # Make y increase downward like array indexing
