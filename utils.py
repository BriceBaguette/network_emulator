import itertools
import numpy as np
import heapq

def saveECMP(dist, pred, number_of_routers, pathList):
    max_int_value = np.iinfo(np.int64).max

    for i in range(number_of_routers):
        if dist[i] != max_int_value:
            j = i
            previous = j

            while pred[j] != -1:
                previous = j
                j = pred[j]

            pathExists = any(len(path) > 0 and path[-1][1] == previous for path in pathList)

            if not pathExists:
                pathList[i].append([i, previous, dist[i]])

class IndexedPriorityQueue:
    def __init__(self, size):
        self.queue = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def push(self, task, priority):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.queue, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.queue:
            priority, count, task = heapq.heappop(self.queue)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

def dijkstra(graph, source):
    number_of_routers = len(graph)
    max_int_value = np.iinfo(np.int64).max

    pathList = [[] for _ in range(number_of_routers)]

    dist = [max_int_value] * number_of_routers
    pred = [-1] * number_of_routers
    sptSet = [0] * number_of_routers

    dist[source] = 0

    heap = IndexedPriorityQueue(number_of_routers)
    heap.push(source, 0)

    while heap.queue:
        u, u_dist = heap.pop()

        if sptSet[u]:
            continue

        sptSet[u] = 1

        for v in range(number_of_routers):
            if not sptSet[v] and graph[u][v] and dist[u] != max_int_value:
                alt = dist[u] + graph[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    pred[v] = u
                    heap.push(v, alt)
                    pathList[v] = []
                    saveECMP(dist, pred, number_of_routers, pathList)
                elif alt == dist[v]:
                    pred[v] = u
                    saveECMP(dist, pred, number_of_routers, pathList)

    return pathList
