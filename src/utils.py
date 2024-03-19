import itertools
import numpy as np
import heapq

def saveECMP(dist, pred, number_of_routers, pathList):
    # Get the maximum possible integer value
    max_int_value = np.iinfo(np.int64).max

    # Iterate over each router
    for i in range(number_of_routers):
        # If the distance to the router is not the maximum integer value (i.e., there is a path to the router)
        if dist[i] != max_int_value:
            # Initialize j and previous to the current router
            j = i
            previous = j

            # Follow the predecessor links until the source router is reached
            while pred[j] != -1:
                # Update previous and j to the predecessor of j
                previous = j
                j = pred[j]

            # Check if a path already exists in the path list that ends at the previous router
            pathExists = any(len(path) > 0 and path[-1][1] == previous for path in pathList)

            # If no such path exists, add a new path to the path list
            if not pathExists:
                # Save the shortest paths
                pathList[i].append([i, previous, dist[i]])



class IndexedPriorityQueue:
    def __init__(self, size):
        self.queue = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def push(self, task, priority):
        # Add a new task or update the priority of an existing task
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.queue, entry)

    def remove_task(self, task):
        # Mark an existing task as REMOVED.  Raise KeyError if not found.
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self):
        # Remove and return the lowest priority task. Raise KeyError if empty.
        while self.queue:
            priority, count, task = heapq.heappop(self.queue)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

def dijkstra(graph, source):
    print("Running Dijkstra's algorithm on source router: {}".format(source))
    # Get the number of routers (nodes) in the graph
    number_of_routers = len(graph)
    # Get the maximum possible integer value
    max_int_value = np.iinfo(np.int64).max

    # Initialize an empty path list for each router
    pathList = [[] for _ in range(number_of_routers)]

    # Initialize the distance to each router as the maximum integer value
    dist = [max_int_value] * number_of_routers
    # Initialize the predecessor of each router as -1
    pred = [-1] * number_of_routers
    # Initialize the shortest path tree set as an array of zeros
    sptSet = [0] * number_of_routers

    # Set the distance to the source router as 0
    dist[source] = 0

    # Create an IndexedPriorityQueue and add the source router to it
    heap = IndexedPriorityQueue(number_of_routers)
    heap.push(source, 0)

    # While the queue is not empty
    while heap.queue:
        # Pop the router with the smallest distance
        u, u_dist = heap.pop()

        # If the router has already been visited, skip it
        if sptSet[u]:
            continue

        # Mark the router as visited
        sptSet[u] = 1

        # For each router in the graph
        for v in range(number_of_routers):
            # If the router has not been visited and there is an edge from u to v and the distance to u is not infinity
            if not sptSet[v] and graph[u][v] and dist[u] != max_int_value:
                # Calculate the alternative distance through u
                alt = dist[u] + graph[u][v]
                # If the alternative distance is smaller
                if alt < dist[v]:
                    # Update the distance to v
                    dist[v] = alt
                    # Update the predecessor of v
                    pred[v] = u
                    # Add v to the queue
                    heap.push(v, alt)
                    # Clear the path list for v
                    pathList[v] = []
                    # Save the shortest paths
                    saveECMP(dist, pred, number_of_routers, pathList)
                # If the alternative distance is equal to the current distance
                elif alt == dist[v]:
                    # Update the predecessor of v
                    pred[v] = u
                    # Save the shortest paths
                    saveECMP(dist, pred, number_of_routers, pathList)

    # Return the list of shortest paths
    return pathList