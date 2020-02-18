class SparseDom:
    def __init__(self, N):
        self.pointer = N - 1
        self.dom = [i for i in range(N)]

    def get(self, index):
        return self.dom[index]

    def delete(self, index):
        tmp = self.dom[index]
        self.dom[index] = self.dom[self.pointer]
        self.dom[self.pointer] = tmp
        self.pointer -= 1

    def restore(self, index):
        self.pointer = index

    def assign(self, index):
        tmp = self.dom[0]
        self.dom[0] = self.dom[index]
        self.dom[index] = tmp
        self.pointer = 0
