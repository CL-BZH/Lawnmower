import turtle
import plotly.offline as pyo
import plotly.graph_objs as go
from maze_conf import *
import time
#import numpy as np
import matplotlib.pyplot as plt
import math



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
        self.mowed = []
        self.distance = -1
        self.efficiency =[]
        self.direction_changes = 0
        self.Qmatrix = None
            
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
        
    def moveTurtle(self, col, row):
        self.current_pos = (row, col)
        x = col
        y = -row
        x += self.xTranslate
        y += self.yTranslate
        self.t.penup()
        self.t.setheading(self.t.towards(x, y))
        self.t.goto(x, y)
    
    def updatePosition(self, row, col, manual=False):
        if self.mazelist[row][col] == OBSTACLE:
            print(f"There is an obstacle in row {row}, column {col}")
            if manual == False:
                exit(-1)
            else:
                return 0
        else:
            self.moveTurtle(col, row)
            return 1
    
    def getDirections(self, row, col):
        """
        Get the row and column for the East, North-East, North, North-West, West, South-West, South, South-East
        directions (4 points connectivity)
        """
        return {'E':(row, col+1), 'NE':(row-1, col+1), 'N':(row-1, col),
                'NW':(row-1, col-1), 'W':(row, col-1), 'SW':(row+1, col-1),
                'S':(row+1, col), 'SE':(row+1, col+1)}
            
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
        adjency_dict = {}

        for vertex, color in zip(vertices, vertices_colors):
            directions = self.getDirections(*vertex)
            neighbors = []
            for compass, neighbor in directions.items():
                weight = 1
                if neighbor in vertices:
                    if compass == 'NE':
                        weight = 2
                        if directions['N'] not in vertices or directions['E'] not in vertices:
                            continue
                    if compass == 'NW':
                        weight = 2
                        if directions['N'] not in vertices or directions['W'] not in vertices:
                            continue
                    if compass == 'SW':
                        weight = 2
                        if directions['W'] not in vertices or directions['S'] not in vertices:
                            continue
                    if compass == 'SE':
                        weight = 2
                        if directions['E'] not in vertices or directions['S'] not in vertices:
                            continue
                    neighbors.append((neighbor, weight))
            adjency_dict[vertex] = (color, neighbors)
        #print(adjency_dict)
        neighbors = {neighbor[0]:neighbor[1:][0][1] for neighbor in adjency_dict.items()}
        #print(neighbors)
        keys = sorted(neighbors.keys())
        #print(keys)
        size = len(keys)
        adjency_mat = [ [0]*size for i in range(size) ]
        for a,b,w in [(keys.index(a), keys.index(b), w) for (a, tmp) in neighbors.items() for b,w in tmp]:
            #print(f"a={a}, b={b}")
            adjency_mat[a][b] = 0 if (a==b) else w

        return adjency_dict, adjency_mat, keys

    def getMST(self, g=None, start=None):
        if not g:
            g = self.getGraph()
        adjency_dict, adjency_mat, ordered_vertices = g

        start_index = 0
        if start:
            start_index = ordered_vertices.index(start[0])
            
        size = len(ordered_vertices)
        key = [math.inf] * size
        parent = [None] * size  # Array to store constructed MST
        children = [[]] * size
        key[start_index] = 0 # To be picked first
        mstSet = [False] * size
        def minKey(key, size, mstSet):
            # Initialize min value
            min = math.inf
            min_index = 0
            
            for vertex_idx in range(size):
                if key[vertex_idx] < min and mstSet[vertex_idx] == False:
                    min = key[vertex_idx]
                    min_index = vertex_idx
 
            return min_index
 
        for _ in range(size):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = minKey(key, size, mstSet)
 
            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(size):
                adjency_mat_u_v = adjency_mat[u][v]
                # Update the key only if adjency_mat[u][v] is smaller than key[v]
                if adjency_mat_u_v > 0 and mstSet[v] == False and key[v] > adjency_mat_u_v:
                    key[v] = adjency_mat_u_v
                    parent[v] = u

        for v in range(size):
            for u in range(size):
                if parent[v] == u:
                    # v is a child of u
                    if children[u] == []:
                        children[u] = [v]
                    else:
                        children[u].append(v)

        print(children)
        return parent, children

    def printMST(self, mst, adjency_mat, key, start_index=None):
        size = len(key)
        if not start_index:
            start_index = [0]

        parent = mst[0]
        print("Parent - Child \tWeight")
        for i in range(size):
            if i == start_index[0]:
                parent_vertex = '*'
                w = 'inf'
            else:
                parent_vertex = key[parent[i]]
                w = adjency_mat[i][parent[i]]
            print(f"{parent_vertex} - {key[i]}\t{w}")

        children = mst[1]
        for i in range(size):
            child_vertices = [key[child] for child in children[i]]
            print(f"{key[i]} -> {child_vertices}")

            
    def drawGraph(self, g=None, tree=None):
        if not g:
            g = self.getGraph()
        adjency_dict, adjency_mat, key = g

        start = (self.startRow, self.startCol)

        if not tree:
            parent, children = self.getMST(g, [start])
        else:
            parent, children = tree
            
        start_index = key.index(start)
        size = len(key)
        tree = []
        for i in range(size):
            if i == start_index:
                continue
            else:
                parent_vertex = key[parent[i]]
                #w = adjency_mat[i][parent[i]]
            tree.append((key[i], parent_vertex))
            
        path = self.path.values()
        traces = []
        for vertex in adjency_dict:
            vx = vertex[1]
            vy = vertex[0]
            annotation = ''
            #v = go.Scatter(x = vx, y = vy, mode = 'markers', name = vertex)
            color = adjency_dict[vertex][0]
            edges_and_weights = adjency_dict[vertex][1:][0]
            #print(f"edges_and_weights: {edges_and_weights}")
            edges = [v for v, w in edges_and_weights]
            #print(f"edges: {edges}")
            if len(edges) > 0:
                for edge in edges:
                    X = [vx]
                    Y = [vy]
                    x = edge[1]
                    y = edge[0]
                    edge_color = adjency_dict[edge][0]
                    if ((edge_color == colors[DEAD_END]) or
                        (color == colors[DEAD_END])):
                        edge_color = colors[DEAD_END]
                    elif (((vertex, edge) in path) or
                          ((edge, vertex) in path)):
                        edge_color = colors[PART_OF_PATH]
                    else:
                        edge_color = colors[NOT_VISITED]
                        if (vertex, edge) in tree or (edge, vertex) in tree:
                            edge_color = 'blue'
                            
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
            self.t.color(self.turtleColor)
            self.t.fillcolor(self.turtleFillColor)
            self.t.setheading(heading)
            self.t.showturtle()
            self.wn.update()
            self.wn.tracer(1)
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
        
    def done(self):
        self.t.color('lime')
        pos = self.t.pos()
        self.t.hideturtle()
        self.t.setpos(0,0)
        self.t.write('Done!', font=("Arial", 20, "bold"))
        self.t.setpos(*pos)
        self.t.color(self.turtleColor)
        self.t.showturtle()
        # Print the efficiency remove 1 from len(self.lawn) since the starting
        # cell is always mowed
        print(f"Global efficiency: {100*(len(self.lawn)-1)/self.distance:.1f}%")
        print(f"Number of direction change: {self.direction_changes}")
            
    def run(self, searchFunction=None):
            
        if searchFunction==None:
            self.mow(*self.current_pos)
            heading = self.t.heading()
            def forward():
                nonlocal heading
                self.wn.onkey(None, 'm')
                current_row, current_col = self.current_pos
                offsets = {0.0: (0, 1), 45.0: (-1,1), 90.0: (-1,0), 135.0: (-1,-1),
                           180.0: (0,-1), 225.0: (1,-1), 270.0: (1,0), 315.0: (1,1)}
                offset = offsets[self.t.heading()]
                row = current_row + offset[0]
                col = current_col + offset[1]
                #print(f"row: {row}, col: {col}")
                if self.updatePosition(row, col, manual=True):
                    self.mow(row, col)
                    if self.t.heading() != heading:
                        heading = self.t.heading()
                        self.direction_changes += 1
                if len(self.mowed) == len(self.lawn):
                    self.wn.onkey(None, 'm')
                    self.wn.onkey(None, 'Left')
                    self.wn.onkey(None, 'Right')
                    self.done()
                else:
                    self.wn.onkeyrelease(lambda: forward(), 'm')
                
            print(f"Your time to play")
            self.wn.onkey(lambda: self.t.setheading((self.t.heading()+45)%360), 'Left')
            self.wn.onkey(lambda: self.t.setheading((self.t.heading()-45)%360), 'Right')
            self.wn.onkeyrelease(lambda: forward(), 'm')
                
            self.wn.listen()
            self.wn.exitonclick()
            #self.wn.mainloop()
        else:
            self.mow(*self.current_pos)
            # Run the algo
            for x, y in searchFunction(self.current_pos[1], -self.current_pos[0],
                                       round(self.t.heading()), self.lawn, self):
                if (x, y) == (math.inf, 0):
                    # this is the code return by the algo to say there is no path to some unmowed lawn
                    print(f"Cannot access unmowed lawn")
                    efficiency = self.efficiency[-1]
                    print(f"Global efficiency: {100*efficiency[0]/efficiency[1]}%")
                    print(f"Number of direction change: {self.direction_changes}")
                    break

                col = x
                row = -y
                current_row, current_col = self.current_pos
                delta_row = abs(current_row - row)
                delta_col = abs(current_col - col)
                if delta_row > 1 or delta_col > 1:
                    print(f"BUG: current row: {current_row}, current_col: {current_col}).\n"
                          f"next row: {row}, next col: {col}).\n" 
                          f"delta_row = {delta_row}, delta_col = {delta_col}")
                    exit(-1)
                if (row != current_row) or (col != current_col):
                    #self.setSpeed(10)
                    current_heading = self.t.heading()
                    self.updatePosition(row, col)
                    self.mow(*self.current_pos)
                    if current_heading != self.t.heading():
                        self.direction_changes += 1
                else:
                    print("Check your algorithm: no movement")
                    exit(-1)
                if len(self.mowed) == len(self.lawn):
                    self.done()
                    break
                #time.sleep(0.5);
            # Exit on click 
            self.wn.exitonclick()
        #turtle.done()
        
    def __getitem__(self,idx):
        return self.mazelist[idx]
    
