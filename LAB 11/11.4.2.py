from collections import deque
import time

# Queue with list
class QueueList:
    def __init__(self):
        self.q = []

    def enqueue(self, x):
        self.q.append(x)

    def dequeue(self):
        return self.q.pop(0)

    def is_empty(self):
        return not self.q

    def __repr__(self):
        return str(self.q)

# Queue with deque
class QueueDeque:
    def __init__(self):
        self.q = deque()

    def enqueue(self, x):
        self.q.append(x)

    def dequeue(self):
        return self.q.popleft()

    def is_empty(self):
        return not self.q

    def __repr__(self):
        return str(list(self.q))


# Demo
ql, qd = QueueList(), QueueDeque()
for v in [1, 2, 3]:
    ql.enqueue(v)
    qd.enqueue(v)

print("List Queue:", ql, "Dequeue ->", ql.dequeue(), "After:", ql)
print("Deque Queue:", qd, "Dequeue ->", qd.dequeue(), "After:", qd)

# Performance
N = 50000

# List performance
start = time.time()
ql = QueueList()
for i in range(N):
    ql.enqueue(i)
while not ql.is_empty():
    ql.dequeue()
t1 = time.time() - start

# Deque performance
start = time.time()
qd = QueueDeque()
for i in range(N):
    qd.enqueue(i)
while not qd.is_empty():
    qd.dequeue()
t2 = time.time() - start

print(f"\nList time : {t1:.4f}s\nDeque time: {t2:.4f}s")
