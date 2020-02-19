
class PropagationHeap:
    def __init__(self):
        self.size = 0
        self.heapList = []

    def push(self, x):
        pos = self.size
        qos = pos

        self.size += 1

        while pos != 0:
            pos = (pos - 1) // 2
            a = self.heapList[pos]
