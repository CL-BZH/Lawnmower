import time
import random
import math
 
#
def drunkard_walk(x, y, heading, valid_xy):
    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
    while True:
        dx, dy = random.choice(offsets)
        new_x, new_y = x+dx, y+dy
        if (new_x, new_y) in valid_xy:
            x, y = new_x, new_y
        else:
            continue
        yield (x, y)


def random_walk(x, y, heading, valid_xy):
    ''' 
    The difference with the drunkard walk is that we keep track of the cell
    already visited.
    Thus, we try to avoid going in the same cell again unless there is no
    choice (that is, all the cells around us were either already visited or
    not valid).
    '''
    visited = []
    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
    steps = offsets[:]
    while True:
        dx, dy = random.choice(steps)
        new_x, new_y = x+dx, y+dy
        if (new_x, new_y) in valid_xy and (new_x, new_y) not in visited:
            x, y = new_x, new_y
            visited.append((x,y))
        elif (new_x, new_y) in visited:
            if len(steps) > 1:
                #print(f"{(new_x, new_y)} is in visited places {visited}")
                #print(f"Remove {(dx, dy)} from {steps}")
                steps.remove((dx, dy))
                continue
            else:
                # No choice have to go to an already visited place
                x, y = new_x, new_y
                steps = offsets[:]
                #print(f"Have to take step {steps}")
        else:
            #print(f"{(new_x, new_y)} not in {valid_xy}")
            steps.remove((dx, dy))
            if len(steps) == 0:
                steps = offsets[:]
            continue
        yield (x, y)

def random_walk_with_policy(x, y, heading, valid_xy):
    ''' 
    The difference with the random walk is that we try to enforce a policy.
    '''
    visited = []
    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
    angles = {(0,1): 90, (0,-1): 270, (1,0): 0, (1,1): 45,
              (1,-1): 315, (-1,0): 180, (-1,1): 135, (-1,-1): 225}
    weights = {(0,1): 0.125, (0,-1): 0.125, (1,0): 0.125, (1,1): 0.125,
               (1,-1): 0.125, (-1,0): 0.125, (-1,1): 0.125, (-1,-1): 0.125}
    directions = {angles[key]: key for key in angles.keys()}
    steps = offsets[:]
    reset = False
    while True:
        bias = [weights[key]+10 if (angles[key]==heading and reset==False) else
                weights[key] for key in steps]
        reset = False
        #print(f"probs: {bias}")
        #print(f"steps: {steps}")
        #print(f"directions: {directions}")
        dx, dy = random.choices(steps, bias)[0]
        new_x, new_y = x+dx, y+dy
        if (new_x, new_y) in valid_xy and (new_x, new_y) not in visited:
            x, y = new_x, new_y
            visited.append((x,y))
        elif (new_x, new_y) in visited:
            if len(steps) > 1:
                #print(f"{(new_x, new_y)} is in visited places {visited}")
                #print(f"Remove {(dx, dy)} from {steps}")
                steps.remove((dx, dy))
                continue
            else:
                # No choice have to go to an already visited place
                x, y = new_x, new_y
                steps = offsets[:]
                reset = True
                #print(f"Have to take step {steps}")
        else:
            #print(f"{(new_x, new_y)} not in {valid_xy}")
            steps.remove((dx, dy))
            if len(steps) == 0:
                steps = offsets[:]
                reset = True
            continue
        heading = angles[(dx, dy)]
        yield (x, y)

        
def greedy_walk(x, y, heading, valid_xy):
    ''' 
    The difference with the random walk is that we try to enforce a policy.
    '''
    visited = []
    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
    angles = {(0,1): 90, (0,-1): 270, (1,0): 0, (1,1): 45,
              (1,-1): 315, (-1,0): 180, (-1,1): 135, (-1,-1): 225}
    weights = {(0,1): 0.125, (0,-1): 0.125, (1,0): 0.125, (1,1): 0.125,
               (1,-1): 0.125, (-1,0): 0.125, (-1,1): 0.125, (-1,-1): 0.125}
    directions = {angles[key]: key for key in angles.keys()}
    steps = offsets[:]
    reset = False
    max_x = 0
    min_y = 0 #because y is negative
    for point in valid_xy:
        px = point[0]
        py = point[1]
        if px > max_x:
            max_x = px
        if py < min_y:
            min_y = py
    while True:
        sum_queen_view = {}
        for offset in offsets:
            dx = offset[0]
            dy = offset[1]
            mov_x = x
            mov_y = y
            view_sum = 0
            distance = 0
            while mov_x > 0 and mov_x <= max_x and mov_y < 0 and mov_y >= min_y:
                mov_x = mov_x + dx
                mov_y = mov_y + dy
                if (mov_x, mov_y) not in valid_xy:
                    break
                if (mov_x, mov_y) not in visited:
                    view_sum = view_sum + 1
                else:
                    if view_sum == 0:
                        distance = distance + 1
            sum_queen_view[offset] = (view_sum, distance)
        best_mov = offsets[0]
        value_max, distance_min = sum_queen_view.get(best_mov)
        sum_queen_view.pop(best_mov)
        
        if (distance_min != 0) or (value_max == 0):
            for mov, mov_factors in sum_queen_view.items():
                distance = mov_factors[1]
                value = mov_factors[0]
                # if value > value_max:
                #     best_mov = mov
                #     value_max = value

                if distance == 0 and value != 0:
                    best_mov = mov
                    value_max = value
                    break
                if distance_min == 0:
                    best_mov = mov
                    value_max = value
                    distance_min = distance
                elif distance > 0 and (value/distance > value_max/distance_min):
                    best_mov = mov
                    value_max = value
                    distance_min = distance


        if value_max != 0:
            while value_max:
                #print(f"best_mov: {best_mov}, value_max = {value_max}")
                x, y = x + best_mov[0], y + best_mov[1]
                if (x, y) not in valid_xy:
                    print("BUG!")
                    exit(-1)
                if (x, y) not in visited:
                    visited.append((x,y))
                    value_max = value_max - 1
                yield (x,y)    
        else:
            # Whatever the direction we do not have direct access
            # to unvisited places. That is we can go to an unvisited
            # place following only one of the queen view.
            # Let's use a A* algorithm to guide us.

            # Select an unvisited position
            for target in valid_xy:
                if target not in visited:
                    break
            # Let's move to that target point
            #print(f"from {(x, y)} to {target}")
            path = a_star_algorithm((x,y), target, valid_xy, offsets)
            for (x,y) in path[1:]:
                if (x, y) not in visited:
                    visited.append((x,y))
                yield (x,y)

            
def a_star_algorithm(start, stop, valid_nodes, offsets):

    h = {node:1 for node in valid_nodes}
    #g = {node:0 for node in valid_nodes}
    #f = {node:0 for node in valid_nodes}
    neighbors = {}
    for node in valid_nodes:
        node_potential_neighbors = [(node[0] + offset[0], node[1] + offset[1])
                                    for offset in offsets]
        #print(potential_neighbors)
    
        node_neighbors = [node_neighbor for node_neighbor in
                          node_potential_neighbors if (node_neighbor
                                                       in valid_nodes)]
        neighbors[node] = node_neighbors
    
    # List of visited nodes with unvisited neighbor(s)
    open_lst = set([start])
    # List of nodes which have been visited and also their neighbor
    closed_lst = set([]) 
 
    # poo has present distances from start to all other nodes
    # the default value is +infinity
    poo = {}
    poo[start] = 0
 
    # par contains an adjac mapping of all nodes
    par = {}
    par[start] = start
 
    while len(open_lst) > 0:
        n = None
 
        # it will find a node with the lowest value of f() -
        for v in open_lst:
            if n == None or poo[v] + h[v] < poo[n] + h[n]:
                n = v;
 
            if n == None:
                print('Path does not exist!')
                return None
 
        # if the current node is the stop
        # then we start again from start
        if n == stop:
            reconst_path = []
 
            while par[n] != n:
                reconst_path.append(n)
                n = par[n]
 
            reconst_path.append(start)
 
            reconst_path.reverse()
 
            #print(f"Path found: {reconst_path}")
            return reconst_path
 
        # for all the neighbors of the current node do
        for m in neighbors[n]:
            weight = 1
            # if the current node is not presentin both open_lst and closed_lst
            # add it to open_lst and note n as it's par
            if m not in open_lst and m not in closed_lst:
                open_lst.add(m)
                par[m] = n
                poo[m] = poo[n] + weight
 
            # otherwise, check if it's quicker to first visit n, then m
            # and if it is, update par data and poo data
            # and if the node was in the closed_lst, move it to open_lst
            else:
                if poo[m] > poo[n] + weight:
                    poo[m] = poo[n] + weight
                    par[m] = n
 
                    if m in closed_lst:
                        closed_lst.remove(m)
                        open_lst.add(m)
                            
        # remove n from the open_lst, and add it to closed_lst
        # because all of his neighbors were inspected
        open_lst.remove(n)
        closed_lst.add(n)
 
    print('Path does not exist!')
    return None
