def compute_ratios(values):
    results = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            denom = values[j] - values[i]
            if denom == 0:
                continue
            ratio = values[i] / denom
            results.append((i, j, ratio))
    return results

nums = [5, 10, 15, 20, 25]
print(compute_ratios(nums))