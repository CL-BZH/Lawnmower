import time
from timeit import default_timer as timer
import random
import math
from queue import PriorityQueue


#
def drunkard_walk(x, y, heading, valid_xy, graph=None):
    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
    while True:
        dx, dy = random.choice(offsets)
        new_x, new_y = x+dx, y+dy
        if (new_x, new_y) in valid_xy:
            x, y = new_x, new_y
        else:
            continue
        yield (x, y)


def random_walk(x, y, heading, valid_xy, graph=None):
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

def random_walk_with_policy(x, y, heading, valid_xy, graph=None):
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

        dx, dy = random.choices(steps, bias)[0]
        new_x, new_y = x+dx, y+dy
        if (new_x, new_y) in valid_xy and (new_x, new_y) not in visited:
            x, y = new_x, new_y
            visited.append((x,y))
        elif (new_x, new_y) in visited:
            if len(steps) > 1:
                steps.remove((dx, dy))
                continue
            else:
                # No choice have to go to an already visited place
                x, y = new_x, new_y
                steps = offsets[:]
                reset = True
        else:
            steps.remove((dx, dy))
            if len(steps) == 0:
                steps = offsets[:]
                reset = True
            continue
        heading = angles[(dx, dy)]
        yield (x, y)

def mc_greedy_learning(start_x, start_y, heading, valid_xy, graph=None):
    print(f"Number of cell to mow: {len(valid_xy)}")
    epochs = 5*len(valid_xy)
    paths = PriorityQueue()
    min_path_len = math.inf
    while epochs:
        print(f"epochs = {epochs}")
        visited = [(start_x, start_y)]
        path = []
        for (x, y) in greedy_walk(start_x, start_y, heading, valid_xy, learn=True):
            if (x, y) not in visited:
                visited.append((x,y))
            path.append((x,y))
            if len(visited) == len(valid_xy):
                #print(f"Full path: {path}")
                break
        if any(path in item for item in paths.queue):
            print(f"Path {path} already in")
        else:
            path_len = len(path)
            if path_len < min_path_len:
                min_path_len = path_len
            paths.put((path_len, path))
        epochs -= 1

    print(f"Minimum Path length: {min_path_len }")
    path = paths.get()[1]
    print(f"Selected Path length: {len(path)}")
    for node in path:
        yield node
    
def greedy_walk(x, y, heading, valid_xy, learn=False, graph=None):
    ''' 
    The difference with the random walk is that we try to enforce a policy.
    '''
    visited = [(x,y)]
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
            view_visited = 0
            distance = math.inf
            sum_dx = dx
            sum_dy = dy
            while mov_x > 0 and mov_x <= max_x and mov_y < 0 and mov_y >= min_y:
                mov_x = x + sum_dx
                mov_y = y + sum_dy
                if ((mov_x, mov_y) not in valid_xy) or \
                   ((offset == (1,1)) and (((x+1, y) not in valid_xy) or ((x, y+1) not in valid_xy))) or \
                   ((offset == (1,-1)) and (((x+1, y) not in valid_xy) or ((x, y-1) not in valid_xy))) or \
                   ((offset == (-1,1)) and (((x-1, y) not in valid_xy) or ((x, y+1) not in valid_xy))) or \
                   ((offset == (-1,-1)) and (((x-1, y) not in valid_xy) or ((x, y-1) not in valid_xy))):
                    #print(f"offset: {offset}. (x,y)={(x, y)}, view_sum={view_sum}")
                    break
                if (mov_x, mov_y) not in visited:
                    if view_sum == 0:
                        # First unvisited cell for the current direction
                        if (sum_dx != dx or sum_dy != dy) and view_visited == 0:
                            print(f"Bug 1: \n\tsum_dx = {sum_dx}\tsum_dy = {sum_dy}"
                                  f"\n\tdx = {dx}\tdy = {dy}")
                        distance = sum_dx**2 + sum_dy**2
                    view_sum = view_sum + 1
                else:
                    view_visited += 1
                    if distance == math.inf:
                        # The first cell in the current direction has already been visited
                        if sum_dx != dx or sum_dy != dy:
                            print(f"Bug 2: \n\tsum_dx = {sum_dx}\tsum_dy = {sum_dy}"
                                  f"\n\tdx = {dx}\tdy = {dy}")
                        distance = sum_dx**2 + sum_dy**2
                    elif view_sum == 0:
                        distance = distance + dx**2 + dy**2
                sum_dx += dx
                sum_dy += dy
                
            if view_sum == 0:
                distance = math.inf
            div = view_sum + view_visited
            if div != 0:
                value = view_sum / div
            else:
                value = 0

            sum_queen_view[offset] = (value, view_sum, distance)

        if learn == True:
            # Randomly select a valid direction
            # d = math.inf
            # offset_candidates = offsets[:]
            # while d == math.inf and offset_candidates != []:
            #     best_mov = random.sample(offset_candidates, 1)[0]
            #     #print(f"Random selection for best mov: {best_mov}")
            #     _, view_sum_max, d = sum_queen_view.get(best_mov)
            #     offset_candidates.remove(best_mov)
            
            offset_candidates = []
            candidates_weights = []
            for mov, (value, view_sum, distance) in sum_queen_view.items():
                offset_candidates.append(mov)
                candidates_weights.append(value)

            best_mov = random.choices(population=offset_candidates,
                                      weights=candidates_weights,
                                      k=1)[0]
            _, view_sum_max, d = sum_queen_view.get(best_mov)
            
        else:
            best_mov = offsets[0]
            value_max, view_sum_max, distance_min = sum_queen_view.get(best_mov)
            sum_queen_view.pop(best_mov)

            #print(f"{(x,y)} -> candidate mov: {best_mov}, distance_min = {distance_min}")
        
            for mov, mov_factors in sum_queen_view.items():
                distance = mov_factors[2]
                view_sum = mov_factors[1]
                value = mov_factors[0]
                
                # if distance == 1 and view_sum == 1:
                #     #Stand alone cell. Select it in order to not leave it behind
                #     best_mov = mov
                #     view_sum_max = view_sum
                #     print(f"Stand alone cell {(x + best_mov[0], y + best_mov[1])} selected")
                #     break
                
                if distance < distance_min:
                    best_mov = mov
                    value_max = value
                    view_sum_max = view_sum
                    distance_min = distance
                elif distance != math.inf and distance >= distance_min:
                    # print(f"\t{mov} distance {distance} >= {distance_min} distance_min")
                    # print(f"\t{mov} value {value} vs {value_max}")
                    ratio = value/distance
                    min_distance_ratio = value_max/distance_min
                    factor = 3
                    if distance == distance_min:
                        factor = 1
                    if (ratio > factor * min_distance_ratio):
                        print(f"ratio {ratio} > {3*min_distance_ratio} 3*min_distance_ratio")
                        best_mov = mov
                        value_max = value
                        view_sum_max = view_sum
                        distance_min = distance

                #print(f"{(x,y)} -> best mov: {best_mov}, distance_min = {distance_min}")
        
        if view_sum_max != 0:
            while view_sum_max:
                x, y = x + best_mov[0], y + best_mov[1]
                if (x, y) not in valid_xy:
                       print(f"BUG! {(x, y)} not in place to mow...")
                       exit(-1)
                if (x, y) not in visited:
                       visited.append((x,y))
                       view_sum_max -= 1
                yield (x,y)
        else:
            # Whatever the direction we do not have direct access
            # to unvisited places. That is we can go to an unvisited
            # place following only one of the queen view.
            # Let's use a A* algorithm to guide us.
            paths = []
            
            # Select an unvisited position
            potential_target = PriorityQueue()
            for target in valid_xy:
                if target not in visited:
                    # Find the "closest" one (considering simply the euclidian distance)
                    potential_target.put((((x-target[0])**2 + (y-target[1])**2), target))

            # Let's move to a target point
            astar_paths = []
            while not potential_target.empty():
                target = potential_target.get()[1]
                astar_path = a_star_algorithm((x,y), target, valid_xy, offsets)
                if astar_path is None:
                    #print(f"astar_path is None for target {target}")
                    continue
                paths.append(astar_path)
                if learn == False:
                    break
                elif len(paths) > 40:
                    # We have to limit the search otherwise it takes too long
                    print("Enforce limitation on call to A*")
                    break

            if paths:
                astar_path = random.sample(paths, 1)[0]
                #print(f"astar_path for target {target}: {astar_path}")
                for x, y in astar_path[1:]:
                    if (x, y) not in visited:
                        visited.append((x,y))
                    yield (x,y)
            else:
                #print(f"No shortest path")
                yield (math.inf, 0)


def get_nodes_neighbors(valid_nodes, offsets):

    neighbors = {}
    for node in valid_nodes:
        node_neighbors = []
        for offset in offsets:
            neighbor_node = (node[0] + offset[0], node[1] + offset[1])
            if (((offset == (1,1)) and
                 (((node[0]+1, node[1]) not in valid_nodes) or ((node[0], node[1]+1) not in valid_nodes))) or \
                ((offset == (1,-1)) and
                 (((node[0]+1, node[1]) not in valid_nodes) or ((node[0], node[1]-1) not in valid_nodes))) or \
                ((offset == (-1,1)) and
                 (((node[0]-1, node[1]) not in valid_nodes) or ((node[0], node[1]+1) not in valid_nodes))) or \
                ((offset == (-1,-1)) and
                 (((node[0]-1, node[1]) not in valid_nodes) or ((node[0], node[1]-1) not in valid_nodes)))):
                # Make sure we do not hit a corner
                    continue
            elif (neighbor_node in valid_nodes):
                node_neighbors.append(neighbor_node)

        if node_neighbors == []:
            print(f"BUG node_neighbors == [] for node {node}")
            exit(1)
        neighbors[node] = node_neighbors

    return neighbors

def a_star_algorithm(start, stop, valid_nodes, offsets):
    #print(f"Run A*")
    h = {node:(start[0] - node[0])**2 + (start[1] - node[1])**2 for node in valid_nodes}

    neighbors = get_nodes_neighbors(valid_nodes, offsets)
    
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

        #print(f"Open list: {open_lst}")
        
        # it will find a node with the lowest value of f() -
        for v in open_lst:
            if n == None or poo[v] + h[v] < poo[n] + h[n]:
                n = v;
 
            if n == None:
                #print('Path does not exist!')
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
            # if the current node is not present in both open_lst and closed_lst
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
 
    print('***Path does not exist!')
    return None



def hamilton_path_util(path, graph, vertices, cycle=False):
    # base case: if all vertices are included in the path
    if len(path) == vertices:
        if cycle == True:
            neighbors = graph[path[-1]][1:][0]
            neighbors = [neighbor for neighbor, weight in neighbors]
            #print(f"neighbors of {path[-1]} are {neighbors}")
            if path[0] not in neighbors:
                print(f"path: {path}")
                return []
        return path

    #print(f"Path: {path}")

    neighbors = graph[path[-1]][1:][0]
    if neighbors == []:
        print(f"BUG: neighbors list for {path[-1]} is empty")
        exit(1)

    closest_neighbors = [neighbor for neighbor, weight in neighbors if weight == 1 and neighbor not in path]
    #print(f"closest neighbors: {closest_neighbors}")
    
    if closest_neighbors == []:
        #print("Take diag neighbors")
        closest_neighbors = [neighbor for neighbor, weight in neighbors]

    if closest_neighbors == []:
        print(f"++++BUG: neighbors list for {path[-1]} is empty")
        exit(1)

    # Shuffle of neighbors list
    random.shuffle(closest_neighbors)
    
    for neighbor in closest_neighbors:
        if neighbor not in path:
            path.append(neighbor)
            if hamilton_path_util(path, graph, vertices, cycle) != []:
                return path

            path.remove(path[-1])

    #print("No solution. Return empty path")
    return []

    
def hamilton_path(start_x, start_y, heading, valid_xy, graph):

    start = timer()
    
    path = [(-start_y, start_x)]
    
    if hamilton_path_util(path, graph, len(valid_xy), cycle=True) == []:
        print ("Solution does not exist\n")
        return []

    #print(f"Path to return: {path[1::]}")
    end = timer()
    
    print(f"Time consumed to find hamilton path: {end - start} seconds")
    
    for (x, y) in path[1::]:
        #print(f"yield {(-y, x)}")
        yield (y, -x)
        


