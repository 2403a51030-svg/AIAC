class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        def _ins(r, k):
            if not r:
                return Node(k)
            if k < r.key:
                r.left = _ins(r.left, k)
            elif k > r.key:
                r.right = _ins(r.right, k)
            return r
        self.root = _ins(self.root, key)

    def search(self, key):
        r = self.root
        while r:
            if key == r.key:
                return True
            r = r.left if key < r.key else r.right
        return False

    def inorder_traversal(self):
        res = []
        def _in(r):
            if r:
                _in(r.left)
                res.append(r.key)
                _in(r.right)
        _in(self.root)
        return res


# Demo
bst = BST()
for x in [50, 30, 70, 20, 40, 60, 80]:
    bst.insert(x)

print("Inorder Traversal:", bst.inorder_traversal())
print("Search 40:", bst.search(40))
print("Search 90:", bst.search(90))
