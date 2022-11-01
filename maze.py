import turtle
import plotly.offline as pyo
import plotly.graph_objs as go
from maze_conf import *
import time
import numpy as np
import matplotlib.pyplot as plt


#turtle.mode('logo')

class Maze:
    def __init__(self, mazeFileName):
        self.mazeFileNane = mazeFileName
        self.graph = []
        rowsInMaze = 0
        columnsInMaze = 0
        self.mazelist = []
        mazeFile = open(mazeFileName,'r')
        rowsInMaze = 0
        self.step = 0
        self.path = {}
        self.edges = {}
        self.startSet = False
        self.endSet = False
        self.startRow = None
        self.startCol = None
        self.endRow = None
        self.endCol = None
        self.lawn = []
        self.current_pos = ()
        self.speed = 3
        for line in mazeFile:
            rowList = []
            col = 0
            for ch in line[:-1]:
                rowList.append(ch)
                if ch == 'L':
                    self.lawn.append((col, -rowsInMaze))
                if ch == 'S':
                    self.startRow = rowsInMaze
                    self.startCol = col
                if ch == 'E':
                    self.endRow = rowsInMaze
                    self.endCol = col
                col = col + 1
            rowsInMaze = rowsInMaze + 1
            self.mazelist.append(rowList)
            columnsInMaze = len(rowList)

        self.turtleFillColor = colors['TURTLE_FILL']
        self.turtleColor = colors['TURTLE']
        self.rowsInMaze = rowsInMaze
        self.columnsInMaze = columnsInMaze
        self.xTranslate = -columnsInMaze/2
        self.yTranslate = rowsInMaze/2
        self.position = turtle.Turtle()
        self.position.shape('turtle')
        self.anchor = turtle.Turtle()
        #self.anchor.shape('turtle')
        self.t = turtle.Turtle()
        turtle.mode('standard') #towards Est
        self.t.shape('turtle')
        self.wn = turtle.Screen()
        self.wn.title("Search Path Turtle")
        self.wn.setup(MAZE_WIDTH + WIDTH_MARGIN, MAZE_HEIGHT + HEIGHT_MARGIN)
        self.wn.screensize(MAZE_WIDTH, MAZE_HEIGHT)
        self.wn.setworldcoordinates(-(columnsInMaze-1)/2-.5,-(rowsInMaze-1)/2-.5,
                                    (columnsInMaze-1)/2+.5,(rowsInMaze-1)/2+.5)
        
        self.t.hideturtle()
        self.t.color(self.turtleColor)
        self.t.fillcolor(self.turtleFillColor)
        self.position.hideturtle()
        self.position.color(self.turtleColor)
        self.position.fillcolor(self.turtleFillColor)
        self.cost = 1 #Cost of a move
        self.values = None
        self.policy = None
        self.mowed = []
        self.distance = -1
        self.efficiency =[]

        
    def setValues(self):
        self.values = np.ones_like(self.mazelist, dtype='uint8') * np.uint8(-1)
        self.policy = np.zeros_like(self.mazelist, dtype='str')
        self.policy[self.endRow, self.endCol] = '*'
        self.values[self.endRow, self.endCol] = 0
        # Get an array of each position inside the maze walls
        self.done = np.zeros_like(self.values[1:-1,1:-1])
        # Set the 'End' position as 'done'
        end_node = (self.endRow, self.endCol)
        self.done[self.endRow - 1, self.endCol - 1] = 1
        # Compute the value of each position
        self.computeValue(end_node)
        print(self.values)
        print(self.policy)

    def computeValue(self, node):
        min = self.values[node]
        node_val = min
        policy = ''
        #print(f"Value before: {value}")
        directions = self.getDirections(*node)
        for compass, next in directions.items():
            node_type = self.getNodeType(*next)
            if (node_type != OBSTACLE):
                val = self.values[next]
                if val < min:
                    min = val
                    policy = compass 
        if min < node_val:
            val = min + self.cost
            self.values[node] = val
            self.policy[node] = policy
            # Update the value of neighbors if needed
            directions = self.getDirections(*node)
            for _, neighbor in directions.items():
                neighbor_type = self.getNodeType(*neighbor)
                if ((neighbor_type != OBSTACLE) and
                    (self.values[neighbor] > val + self.cost)):
                    self.done[neighbor[0]-1, neighbor[1]-1] = 0

        directions = self.getDirections(*node)
        for _, next in directions.items():
            node_type = self.getNodeType(*next)
            if (node_type != OBSTACLE):
                i, j = next[0]-1, next[1]-1
                if (self.done[i, j] == 0):
                    self.done[i, j] = 1
                    self.computeValue(next)

    def getValue(self, node):
        return self.values[node]

    def getPolicy(self, node):
        return self.policy[node]
            
    def drawMaze(self):
        self.anchor.hideturtle()
        self.t.speed(10)
        self.wn.tracer(0)
        for r in range(self.rowsInMaze):
            for c in range(self.columnsInMaze):
                if self.mazelist[r][c] == OBSTACLE:
                    self.drawCenteredBox(c+self.xTranslate,
                                         -r+self.yTranslate,
                                         colors[OBSTACLE])
                else:
                    self.drawCenteredBox(c+self.xTranslate,
                                         -r+self.yTranslate,
                                         colors[LAWN],
                                         colors['GRID'])
        self.wn.update()
        self.wn.tracer(1)
        if (self.startRow is not None) and (self.startCol is not None):
            self.setStart(self.startRow, self.startCol)
        if (self.endRow is not None) and (self.endCol is not None):
            self.setEnd(self.endRow, self.endCol)
        
    def setStart(self, row, col, heading=None):
        if self.startSet == True:
            print(f"Check {self.mazeFileNane} - Start point"
                  f" already set at row {self.startRow}, column {self.startCol}")
            exit(-1)
        if self.mazelist[row][col] == OBSTACLE:
            print(f"There is an obstacle in row {row}, column {col}")
            exit(-1)
        self.startSet = True   
        self.startRow = row
        self.startCol = col
        self.current_pos = (row, col)
        # Mark the start point
        self.t.hideturtle()
        self.moveTurtle(self.startCol, self.startRow)
        if heading != None:
            print(f"Set heading to {heading}")
            self.t.setheading(heading)
        self.t.color(colors[START])
        #self.t.write('Start', font=("Arial", 16, "bold"))
        self.t.color(self.turtleColor)
        self.t.fillcolor(self.turtleFillColor)
        self.t.showturtle()
        time.sleep(2)
        
    def setEnd(self, row, col):
        if self.endSet == True:
            print(f"Check {self.mazeFileNane} - End point"
                  f" already set at row {self.endRow}, column {self.endCol}")
            exit(-1)
        if self.mazelist[row][col] == OBSTACLE:
            print(f"There is an obstacle in row {row}, column {col}")
            exit(-1)
        self.endSet = True
        self.endRow = row
        self.endCol = col
        # Mark the end point
        self.t.hideturtle()
        self.moveTurtle(self.endCol, self.endRow)
        self.t.color(colors[END])
        self.t.dot(10)
        self.t.write('End', font=("Arial", 16, "bold"))
        # Go back to the start
        if (self.startRow is not None) and (self.startCol is not None):
            self.moveTurtle(self.startCol, self.startRow)        
        self.t.color(self.turtleColor)
        self.t.fillcolor(self.turtleFillColor)
        self.t.showturtle()

    def setSpeed(self, speed):
        self.speed = speed
        self.t.speed(speed)
    
    def drawCenteredBox(self, x, y, fillcolor, border_color=None):
        self.t.up()
        self.t.goto(x-.5,y-.5)
        self.t.fillcolor(fillcolor)
        self.t.pencolor(fillcolor)
        if border_color:
            self.t.pencolor(border_color)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(1)
            self.t.right(90)
        self.t.end_fill()

    def showPosition(self, row, col, turtleColor=None, turtleFillColor=None):
        x = col
        y = -row
        x += self.xTranslate
        y += self.yTranslate
        self.position.hideturtle()
        if turtleColor is None:
            self.position.color(self.turtleColor)
        else:
            self.position.color(turtleColor)
        if turtleFillColor is None:
            self.position.fillcolor(self.turtleFillColor)
        else:
            self.position.fillcolor(turtleFillColor)
        self.position.penup()
        self.position.setheading(90) #self.position.towards(x, y))
        self.position.goto(x, y)
        self.position.showturtle()

    def hidePosition(self):
        self.position.hideturtle()
        
    def moveTurtle(self, col, row):
        self.current_pos = (row, col)
        x = col
        y = -row
        x += self.xTranslate
        y += self.yTranslate
        self.t.penup()
        self.t.setheading(self.t.towards(x, y))
        self.t.goto(x, y)

    def dropBreadcrumb(self, color):
        self.t.dot(10, color)
    
    def updatePosition(self, row, col):
        if self.mazelist[row][col] == OBSTACLE:
            print(f"There is an obstacle in row {row}, column {col}")
            exit(-1)
        else:
            self.moveTurtle(col, row)
           
    def isEnd(self, row, col):
        return (row == self.endRow and col == self.endCol)
           
    def isStart(self, row, col):
        return (row == self.startRow and col == self.startCol)

    def getStartNode(self):
        return (self.startRow, self.startCol)
    
    def getEndNode(self):
        return (self.endRow, self.endCol)
    
    def getDirections(self, row, col):
        """
        Get the row and column for the East, North, West and South
        directions (4 points connectivity)
        """
        return {'E':(row, col+1), 'N':(row-1, col), 'W':(row, col-1), 'S':(row+1, col)}

    def getNeighbor(self, compass, node):
        """
        Get the row and column for the East, North, West and South
        directions (4 points connectivity)
        """
        moves = {'E':(0, 1), 'N':(-1, 0), 'W':(0, -1), 'S':(1, 0)}
        mov = moves[compass]
        return [n-m for n, m in zip(node, mov)]

    def getNodeType(self, row, col):
        return self.mazelist[row][col]
    
    def setNodeType(self, row, col, val):
        self.mazelist[row][col] = val

    def showAnchor(self, row, col):
        x = col
        y = -row
        x += self.xTranslate
        y += self.yTranslate
        self.anchor.clear()
        self.anchor.penup()
        self.anchor.goto(x, y)
        #self.anchor.showturtle()
        self.anchor.write('Anchor', font=("Arial", 11, "bold"))

    def hideAnchor(self):
        self.anchor.clear()
        
    def getGraph(self):
        """
        Returns a dictionary, where each key is a node in the graph,
        and the value for a given key is a tuple (color, neighbors)
        where color is the color of the node and neighbors are the 
        neighbor nodes to which it is connected to.
        """
        # First get all vertices
        vertices = []
        vertices_colors = []
        for row in range(self.rowsInMaze):
            for col in range(self.columnsInMaze):
                if self.mazelist[row][col] != OBSTACLE:
                    color = colors[self.mazelist[row][col]]
                    vertices.append((row, col))
                    vertices_colors.append(color)
        # Get all neighbor nodes
        adjency = {}
        for vertex, color in zip(vertices, vertices_colors):
            directions = self.getDirections(*vertex)
            neighbors = []
            for compass, neighbor in directions.items():
                if neighbor in vertices:
                    neighbors.append(neighbor)
            adjency[vertex] = (color, neighbors)
        return adjency

    def drawGraph(self):
        g = self.getGraph()
        path = self.path.values()
        traces = []
        for vertex in g:
            vx = vertex[1]
            vy = vertex[0]
            annotation = ''
            #v = go.Scatter(x = vx, y = vy, mode = 'markers', name = vertex)
            color = g[vertex][0]
            edges = g[vertex][1:][0]
            if len(edges) > 0:
                for edge in edges:
                    X = [vx]
                    Y = [vy]
                    x = edge[1]
                    y = edge[0]
                    edge_color = g[edge][0]
                    if ((edge_color == colors[DEAD_END]) or
                        (color == colors[DEAD_END])):
                        edge_color = colors[DEAD_END]
                    elif (((vertex, edge) in path) or
                          ((edge, vertex) in path)):
                        edge_color = colors[PART_OF_PATH]
                    else:
                        edge_color =  colors[NOT_VISITED]
                    X.append(x)
                    Y.append(y)
                    # Plot the edges
                    traces.append(go.Scatter(x = X, y = Y, mode = 'lines',
                                             marker=dict(size=10,
                                                         color=edge_color),))
            # Plot the vertex
            if vertex == (self.startRow, self.startCol):
                contour = colors[START]
                traces.append(go.Scatter(x = [vx], y = [vy],
                                         mode = 'markers+text',
                                         textposition='top right',
                                         textfont=dict(color='black'),
                                         marker=dict(size=20, color=color,
                                                     line=dict(color=contour,
                                                               width=5)),
                                         name=f"{vertex}", text='Start'))
            elif vertex == (self.endRow, self.endCol):
                contour = colors[END]
                traces.append(go.Scatter(x = [vx], y = [vy],
                                         mode = 'markers+text',
                                         textposition='top right',
                                         textfont=dict(color='black'),
                                         marker=dict(size=20, color=color,
                                                     line=dict(color=contour,
                                                               width=5)),
                                         name=f"{vertex}", text='End'))
            else:
                traces.append(go.Scatter(x = [vx], y = [vy],
                                         mode = 'markers',
                                         marker=dict(size=20, color=color),
                                         name=f"{vertex}"))

            # Plot the policy
            if self.policy is not None:
                policy = self.policy[vertex]
                traces.append(go.Scatter(x = [vx], y = [vy],
                                         mode = 'text',
                                         textposition='middle center',
                                         textfont=dict(family="sans serif",
                                                       size=16, color='black'),
                                         name=f"{vertex}", text=policy))
                
        layout = go.Layout(title = 'Graph',
                           xaxis = dict(showgrid = False),
                           yaxis = dict(showgrid = False, autorange='reversed'),
                           showlegend=False)

        fig = go.Figure(data=traces, layout=layout)
        #pyo.plot(fig, filename='graph.html')
        fig.show()

    def mow(self, row, col):
        self.distance = self.distance + 1
        mowed = False
        if (row, col) not in self.mowed:
            mowed = True
            self.mowed.append((row, col))
            x = col
            y = -row
            heading = self.t.heading()
            #print(f"heading: {self.t.heading()}")
            self.wn.tracer(0)
            self.t.hideturtle()
            self.t.speed(10)
            self.drawCenteredBox(x+self.xTranslate,
                                 y+self.yTranslate,
                                 colors[VISITED])
            self.moveTurtle(col, row)
            self.wn.update()
            self.wn.tracer(1)
            self.t.color(self.turtleColor)
            self.t.fillcolor(self.turtleFillColor)
            self.t.setheading(heading)
            self.t.showturtle()
            self.t.speed(self.speed)
        if self.efficiency == []:
            self.efficiency.append((1,1))
        else:
            eff = self.efficiency[-1]
            self.efficiency.append((eff[0]+mowed, eff[1]+1))
            

    def plotEfficiency(self):
        xpoints = [efficiency[1] for efficiency in self.efficiency]
        ypoints = [100*efficiency[0]/efficiency[1] for efficiency in self.efficiency]

        plt.plot(xpoints, ypoints)
        plt.title("Efficiency")
        plt.xlabel("Number of step")
        plt.ylabel("Percentage of efficiency")
        plt.show()
        
    def run(self, searchFunction=None):
        if searchFunction==None:
            self.wn.exitonclick()
        else:
            self.mow(*self.current_pos)
            # Run the algo
            for x, y in searchFunction(self.current_pos[1],-self.current_pos[0],
                                       round(self.t.heading()), self.lawn):
                col = x
                row = -y
                current_row, current_col = self.current_pos
                delta_row = abs(current_row - row)
                delta_col = abs(current_col - col)
                if delta_row > 1 or delta_col > 1:
                    print(f"BUG: delta_row = {delta_row}, delta_col = {delta_col}")
                    exit(-1)
                if (row != current_row) or (col != current_col):
                    self.setSpeed(10)
                    self.updatePosition(row, col)
                    self.mow(*self.current_pos)
                else:
                    print("Check your algorithm: no movement")
                    exit(-1)
                if len(self.mowed) == len(self.lawn):
                    self.t.color('lime')
                    pos = self.t.pos()
                    self.t.setpos(0,0)
                    self.t.hideturtle()
                    self.t.write('Done!', font=("Arial", 20, "bold"))
                    self.t.setpos(*pos)
                    self.t.color(self.turtleColor)
                    self.t.showturtle()
                    print(f"Global efficiency: {100*len(self.lawn)/self.distance:.1f}%")
                    
                    break
                #time.sleep(0.5);
            # Exit on click 
            self.wn.exitonclick()
        #turtle.done()

    def pausing(self):
        while self.status == 'pause':
            time.sleep(0.2)
            self.wn.update()

    def pause(self):
        if (self.status == 'running'):
            self.status = 'pause'
        else: # (self.tatus == 'pause')
            self.status = 'running'
        print(self.status)
        
    def __getitem__(self,idx):
        return self.mazelist[idx]


