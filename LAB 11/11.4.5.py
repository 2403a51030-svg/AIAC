from collections import deque

class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v):
        # Build adjacency list (undirected graph)
        self.adj.setdefault(u, []).append(v)
        self.adj.setdefault(v, []).append(u)

    def bfs(self, start):
        vis, q, order = set(), deque([start]), []
        while q:
            n = q.popleft()
            if n not in vis:
                vis.add(n)
                order.append(n)
                for x in self.adj.get(n, []):  # ✅ safe lookup
                    if x not in vis:
                        q.append(x)
        return order

    def dfs_recursive(self, start, vis=None):
        if vis is None:
            vis = set()
        vis.add(start)
        order = [start]
        for x in self.adj.get(start, []):  # ✅ safe lookup
            if x not in vis:
                order += self.dfs_recursive(x, vis)
        return order

    def dfs_iterative(self, start):
        vis, stack, order = set(), [start], []
        while stack:
            n = stack.pop()
            if n not in vis:
                vis.add(n)
                order.append(n)
                stack.extend(reversed(self.adj.get(n, [])))  # ✅ safe lookup
        return order


# Demo
g = Graph()
edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("E", "F")]
for u, v in edges:
    g.add_edge(u, v)

print("BFS :", g.bfs("A"))
print("DFS Recursive:", g.dfs_recursive("A"))
print("DFS Iterative:", g.dfs_iterative("A"))
