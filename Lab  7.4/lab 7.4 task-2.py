def sort_list(data):
    # Filter only string elements and sort them
    return sorted([item for item in data if isinstance(item, str)])

items = [3, "apple", 1, "banana", 2]
print(sort_list(items))  # Output: ['apple', 'banana']