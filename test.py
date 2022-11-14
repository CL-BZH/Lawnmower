from maze import Maze
from lawnmower_algo import drunkard_walk
from lawnmower_algo import random_walk
from lawnmower_algo import random_walk_with_policy
from lawnmower_algo import greedy_walk
from lawnmower_algo import mc_greedy_learning
from lawnmower_algo import hamilton_cycle

# Instantiate the maze reading the maze's describtion file
#maze = Maze('maze0.txt')
#maze = Maze('maze1.txt')
#maze = Maze('maze2.txt')
maze = Maze('maze3.txt')
#maze = Maze('maze4.txt')
#maze = Maze('maze5.txt')

# Draw the maze
maze.drawMaze()

# Set the 'Start' point if not set in the text file
# describing the maze
#maze.setStart(row=4, col=10, heading=0)
#maze.setStart(row=10, col=14, heading=0)
#maze.setStart(row=6, col=1, heading=0)
#maze.setStart(row=10, col=1, heading=45)
#maze.setStart(row=1, col=19, heading=180)
#maze.setStart(row=3, col=2, heading=0)
maze.setStart(row=3, col=1, heading=0)


# Set the turtle speed (between 1 and 10) (default to 3)
maze.setSpeed(10)

# Start mowing the lawn
#maze.run(drunkard_walk)
#maze.run(random_walk_with_policy)
#maze.run(greedy_walk)
#maze.run(mc_greedy_learning)
maze.run(hamilton_cycle)

# You enter the move
#maze.run()

# Draw the graph of the maze
#maze.drawGraph()

# Plot efficiency
#maze.plotEfficiency()
