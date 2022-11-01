from maze import Maze
from lawnmower_algo import drunkard_walk
from lawnmower_algo import random_walk
from lawnmower_algo import random_walk_with_policy
from lawnmower_algo import greedy_walk


# Instantiate the maze reading the maze's describtion file
maze = Maze('maze0.txt')
#maze = Maze('maze1.txt')

# Draw the maze
maze.drawMaze()

# Set the 'Start' point if not set in the text file
# describing the maze
#maze.setStart(row=4, col=10, heading=0)
#maze.setStart(row=10, col=14, heading=0)
#maze.setStart(row=6, col=1, heading=0)
#maze.setStart(row=10, col=1, heading=45)
maze.setStart(row=1, col=19, heading=180)

# Set the turtle speed (between 1 and 10) (default to 3)
maze.setSpeed(10)

# Start mowing the lawn
#maze.run(drunkard_walk)
#maze.run(random_walk_with_policy)
maze.run(greedy_walk)

# Draw the graph of the maze
#maze.drawGraph()

# Plot efficiency
maze.plotEfficiency()
