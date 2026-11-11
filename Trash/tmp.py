    # Modify the MST
    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
    neighbors = get_nodes_neighbors(valid_xy, offsets)
    print(neighbors)
    size = len(key)

    max_distance = 0
    farest_leaf_index = key.index(start)
    for u in range(size):
        if children[u] == []:
            u_dist = 0
            v = parent[u]
            # u is the index of a leaf
            while v != key.index(start):
                u_dist += 1
                v = parent[v]
            if u_dist > max_distance:
                max_distance = u_dist
                farest_leaf_index = u
    farest_leaf = key[farest_leaf_index]
    print(f"The farest leaf is {farest_leaf} at distance {max_distance}")
    farest_leaf = (farest_leaf[1], -farest_leaf[0])
    for _ in range(size):
        for u in range(size):
            if children[u] == []:
                # u is a leaf
                # if it is a neighbor of the farest leaf then connect them
                leaf = key[u]
                leaf_neighbors = neighbors[(leaf[1], -leaf[0])]
                #print(f"leaf {leaf} has neighbors {leaf_neighbors}. farest_leaf is {farest_leaf}")

                if farest_leaf in leaf_neighbors:
                    children[farest_leaf_index] = u
                    children[parent[u]].remove(u)
                    parent[u] = farest_leaf_index
                    farest_leaf_index = u
                    farest_leaf = key[farest_leaf_index]
                    farest_leaf = (farest_leaf[1], -farest_leaf[0])
                    print(f"The farest leaf is {farest_leaf} at distance {max_distance+1}")
                
    # for u in range(size):
    #     if children[u] == []:
    #         # u is the index of a leaf
    #         leaf = key[u]
    #         leaf_neighbors = neighbors[(leaf[1], -leaf[0])]
    #         leaf_neighbors = [(-leaf_neighbor[1],leaf_neighbor[0])
    #                           for leaf_neighbor in leaf_neighbors]
    #         print(f"{leaf} has neighbors: {leaf_neighbors}")
    #         for leaf_neighbor in leaf_neighbors:
    #             leaf_neighbor_index = key.index(leaf_neighbor)
    #             if parent[leaf_neighbor_index] is not None: # start has no parent
    #                 print(f"Parent of {leaf_neighbor} (idx:{leaf_neighbor_index}) "
    #                       f"is {key[parent[leaf_neighbor_index]]} "
    #                       f"(idx:{parent[leaf_neighbor_index]})")
                
    #             if (leaf_neighbor_index != parent[u] and
    #                 parent[u] not in children[leaf_neighbor_index]):
    #                 print(f"Parent of {leaf_neighbor} becomes {leaf}")
    #                 parent[leaf_neighbor_index] = u
            
    start_index = key.index(start)

    maze.drawGraph(g, (parent, children))
    #maze.printMST(mst, adjency_mat, key, [start_index])

        def build_path(parent, adjency_mat, key, start, valid_xy):
        index = key.index(start)
        #print(f"index is {index}")
        path = [start]
        size = len(key)
        visited = [start]
        while len(visited) != len(valid_xy):
            next_vertex_idx = parent[index]

            if next_vertex_idx is not None and key[next_vertex_idx] not in path:
                vertex = key[next_vertex_idx]
                visited.append(vertex)
            else:
                # next_vertex_idx is None -> At the tree root
                # key[next_vertex_idx] in path -> end of branch

                # Get all neighbors of current vertex.
                # Select that has the smallest weight
                neighbors_index = adjency_mat[index][:]
                neighbors = [(key[index], neighbors_index[index]) for
                             index in range(size) if neighbors_index[index] != 0]
                neighbors = [(neighbor, weight) if neighbor not in path
                             else (neighbor, 10) for (neighbor, weight) in neighbors]

                #print(f"vertex index {index}. Neighbors: {neighbors}")
                    
                ordered_neighbors = PriorityQueue()
                if len(path) > 1:
                    vertex = path[-1]
                    previous_vertex = path[-2]
                    dir = (abs(vertex[0] - previous_vertex[0]),
                           abs(vertex[1] - previous_vertex[1]))
                    for neighbor, weight in neighbors:
                        ndir = (abs(vertex[0] - neighbor[0]),
                                abs(vertex[1] - neighbor[1]))
                        if ndir == dir:
                            weight -= 2
                        if neighbor in path:
                            weight += 10
                        ordered_neighbors.put((weight, neighbor))
                else:
                    for neighbor, weight in neighbors:
                        if neighbor in path:
                            weight += 10
                        ordered_neighbors.put((weight, neighbor))
                
                vertex = ordered_neighbors.get()[1]
                if vertex in path:
                    #print(f"Vertex {vertex} is already in path {path}")

                    # No choice go to closest neighbor not yet lawned
                    # Select an unvisited position
                    potential_target = PriorityQueue()
                    vertex = path[-1]
                    x = vertex[1]
                    y = -vertex[0]
                    for (vx,vy) in valid_xy:
                        target = (-vy, vx)
                        if target not in visited:
                            # Find the "closest" one (considering simply the euclidian distance)
                            potential_target.put((((x-target[0])**2 + (y-target[1])**2), target))
                    if potential_target.empty():
                        print(f"BUG: No unvisited node found!!!!")
                        exit(1)
                    
                    target = potential_target.get()[1]
                    offsets = [(0,1), (0,-1), (1,0), (1,1), (1,-1), (-1,0), (-1,1), (-1,-1),]
                    (r, c) = path[-1]
                    (x, y) = (c, -r)
                    (r, c) = target
                    (target_x, target_y) = (c, -r)
                    print(f"find path from {(x,y)} to {(target_x, target_y)}")
                    astar_path = a_star_algorithm((x,y), (target_x, target_y),
                                                  valid_xy, offsets)
                    #print(f"A* path: {astar_path}")
                    for (x,y) in astar_path[1:]:
                        vertex = (-y, x)
                        if vertex not in visited:
                            visited.append(vertex)
                        path.append(vertex)
                    # Now we have vertex = path[-1]
                    path = path[:-1]
                else:
                    visited.append(vertex)
            path.append(vertex)
            index = key.index(vertex)
        #print(f"The complete path is {path}")
        return path[1:]
