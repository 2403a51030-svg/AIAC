def linear_search_early_exit(arr, target):
    comparisons = 0
    for i in range(len(arr)):
        comparisons += 1
        if arr[i] == target:
            return i, comparisons  # Early exit if found
    return -1, comparisons  # Not found

# Example list of 20 elements
lst = [3, 7, 8, 5, 2, 9, 1, 4, 0, 6, 12, 14, 13, 10, 11, 19, 18, 16, 17, 15]
target = 14  # Change as you like

index, num_comparisons = linear_search_early_exit(lst, target)
if index != -1:
    print(f"Target {target} found at index {index} after {num_comparisons} comparisons")
else:
    print(f"Target {target} not found after {num_comparisons} comparisons")
