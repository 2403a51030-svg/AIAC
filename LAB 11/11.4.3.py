class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def insert_at_end(self, data):
        n = Node(data)
        if not self.head:
            self.head = n
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = n

    def delete_value(self, val):
        if not self.head:
            return
        if self.head.data == val:
            self.head = self.head.next
            return
        cur = self.head
        while cur.next and cur.next.data != val:
            cur = cur.next
        if cur.next:
            cur.next = cur.next.next
        else:
            print(f"Value {val} not found")

    def traverse(self):
        res, cur = [], self.head
        while cur:
            res.append(cur.data)
            cur = cur.next
        return res


# Demo
ll = LinkedList()
for x in [10, 20, 30]:
    ll.insert_at_end(x)

print("After insertions:", ll.traverse())
ll.delete_value(20)
print("After deleting 20:", ll.traverse())
ll.delete_value(10)
print("After deleting 10:", ll.traverse())
ll.delete_value(40)
ll.delete_value(30)
print("After deleting 30:", ll.traverse())
