import random
import time

# Static stock data
STATIC_STOCKS = [
    {'symbol': 'STK0', 'open': 120.50, 'close': 125.75, 'change_pct': 4.36},
    {'symbol': 'STK1', 'open': 200.00, 'close': 190.00, 'change_pct': -5.00},
    {'symbol': 'STK2', 'open': 50.25, 'close': 51.00, 'change_pct': 1.49},
    {'symbol': 'STK3', 'open': 345.10, 'close': 360.50, 'change_pct': 4.46},
    {'symbol': 'STK4', 'open': 88.00, 'close': 85.50, 'change_pct': -2.84},
    {'symbol': 'STK5', 'open': 450.75, 'close': 451.00, 'change_pct': 0.06},
    {'symbol': 'STK6', 'open': 15.30, 'close': 16.50, 'change_pct': 7.84},
    {'symbol': 'STK7', 'open': 275.00, 'close': 275.00, 'change_pct': 0.00},
    {'symbol': 'STK8', 'open': 102.40, 'close': 98.15, 'change_pct': -4.15},
    {'symbol': 'STK9', 'open': 300.00, 'close': 325.00, 'change_pct': 8.33}
]

# Heap Sort implementation
def heapify(arr, n, i, key):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l][key] > arr[largest][key]:
        largest = l
    if r < n and arr[r][key] > arr[largest][key]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, key)

def heap_sort(arr, key):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, key)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, key)
    arr.reverse()

# Search using Hash Map
def build_stock_map(stocks):
    return {stock['symbol']: stock for stock in stocks}

def search_stock(stock_map, symbol):
    return stock_map.get(symbol, None)

# Performance comparison
def compare_performance(stocks):
    stocks_copy_heap = stocks.copy()
    start_time_heap = time.perf_counter()
    heap_sort(stocks_copy_heap, 'change_pct')
    end_time_heap = time.perf_counter()
    heap_sort_time = end_time_heap - start_time_heap

    stocks_copy_sorted = stocks.copy()
    start_time_sorted = time.perf_counter()
    sorted(stocks_copy_sorted, key=lambda x: x['change_pct'], reverse=True)
    end_time_sorted = time.perf_counter()
    sorted_time = end_time_sorted - start_time_sorted

    stock_map = build_stock_map(stocks)
    search_symbol = 'STK3'

    start_time_linear = time.perf_counter()
    result_linear = next((stock for stock in stocks if stock['symbol'] == search_symbol), None)
    end_time_linear = time.perf_counter()
    linear_search_time = end_time_linear - start_time_linear

    start_time_hash = time.perf_counter()
    result_hash = search_stock(stock_map, search_symbol)
    end_time_hash = time.perf_counter()
    hash_search_time = end_time_hash - start_time_hash

    print(f"Heap Sort time:        {heap_sort_time:.6f}s")
    print(f"Python sorted() time:  {sorted_time:.6f}s")
    print(f"Linear search time:    {linear_search_time:.6f}s")
    print(f"Hash map search time:  {hash_search_time:.6f}s")
    
    print("\n--- Trade-offs ---")
    print("Sorting: Heap Sort is a solid general-purpose algorithm with a worst-case time complexity of O(n log n). Python's built-in `sorted()` is often faster for typical cases due to highly optimized C implementations (Timsort).")
    print("Searching: Hash map lookups have an average time complexity of O(1), making them extremely fast regardless of dataset size. Linear search is O(n), becoming very slow for large datasets.")


if __name__ == "__main__":
    stocks_to_sort = STATIC_STOCKS.copy()
    heap_sort(stocks_to_sort, 'change_pct')
    print("Top 5 stocks by % change (Heap Sort):")
    for stock in stocks_to_sort[:5]:
        print(f"  {stock}")

    print("\n--- Performance Comparison ---")
    compare_performance(STATIC_STOCKS)

    print("\n--- Example Search ---")
    stock_map = build_stock_map(STATIC_STOCKS)
    symbol_to_find = 'STK6'
    result = search_stock(stock_map, symbol_to_find)
    if result:
        print(f"Found stock with symbol '{symbol_to_find}': {result}")
    else:
        print(f"Stock symbol '{symbol_to_find}' not found.")