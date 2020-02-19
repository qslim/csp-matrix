import heapq


class PriorityQueue:
    def __init__(self, N):
        self._queue = []
        self.size = 0
        self.map = [0 for _ in range(N)]

    def push(self, priority, item):
        if self.map[item] == 0:
            self.map[item] = 1
            self.size += 1
            heapq.heappush(self._queue, (priority, item))
        else:
            pass

    def pop(self):
        self.size -= 1
        item = heapq.heappop(self._queue)[-1]
        self.map[item] = 0
        return item

    def empty(self):
        return self.size == 0

    def clear(self):
        for i in range(self.size):
            item = self.pop()
            self.map[item] = 0
        self.size = 0


q = PriorityQueue(4)
q.push(3, 1)
q.push(3, 2)
q.push(4, 3)
q.push(4, 3)

print(q.pop())
print(q.pop())
print(q.pop())
print(q.pop())