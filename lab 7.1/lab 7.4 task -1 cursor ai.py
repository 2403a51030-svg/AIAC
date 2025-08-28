def factr(n):
image.png    # Accept numeric strings and convert to int
    if isinstance(n, str):
        if n.strip().isdigit():
            n = int(n)
        else:
            raise ValueError("Input must be a non-negative integer")
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    if n == 0 or n == 1:
        return 1
    return n * factr(n - 2)

print(factr("5"))  # 15 (double factorial: 5 * 3 * 1)