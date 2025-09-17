import csv
import json
import time

# --- Static Dataset (books.csv as a list of dictionaries) ---
# In a real-world scenario, you would load this from a file.
# For this example, we'll use a hardcoded list to meet the "static input" requirement.
static_books = [
    {'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams'},
    {'title': 'The Restaurant at the End of the Universe', 'author': 'Douglas Adams'},
    {'title': 'Life, the Universe and Everything', 'author': 'Douglas Adams'},
    {'title': 'So Long, and Thanks for All the Fish', 'author': 'Douglas Adams'},
    {'title': 'Mostly Harmless', 'author': 'Douglas Adams'},
    {'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien'},
    {'title': 'The Hobbit', 'author': 'J.R.R. Tolkien'},
    {'title': 'A Brief History of Time', 'author': 'Stephen Hawking'},
    {'title': 'Cosmos', 'author': 'Carl Sagan'},
    {'title': 'Contact', 'author': 'Carl Sagan'},
    {'title': '1984', 'author': 'George Orwell'},
    {'title': 'Animal Farm', 'author': 'George Orwell'},
    {'title': 'Dune', 'author': 'Frank Herbert'},
    {'title': 'The Foundation Trilogy', 'author': 'Isaac Asimov'},
    {'title': 'I, Robot', 'author': 'Isaac Asimov'},
    {'title': 'Fahrenheit 451', 'author': 'Ray Bradbury'},
    {'title': 'Brave New World', 'author': 'Aldous Huxley'},
    {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee'},
    {'title': 'Pride and Prejudice', 'author': 'Jane Austen'},
    {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald'}
]

# --- Search Algorithm Implementations ---

def linear_search(books, keyword):
    """
    Performs a linear search on a list of books.
    Time Complexity: O(n)
    """
    keyword = keyword.lower()
    return [book for book in books if keyword in book['title'].lower() or keyword in book['author'].lower()]

def binary_search_optimized(books, keyword):
    """
    Performs a binary search on a sorted list of books by title.
    Note: A full-text binary search is complex. This is a simplified example
    that works on the sorted titles to demonstrate the principle.
    Time Complexity: O(log n) after initial sort.
    """
    keyword = keyword.lower()
    # Sort the list once for binary search to work efficiently
    books_sorted = sorted(books, key=lambda x: x['title'].lower())
    
    results = []
    left, right = 0, len(books_sorted) - 1
    
    while left <= right:
        mid = (left + right) // 2
        title = books_sorted[mid]['title'].lower()
        
        if keyword == title:
            # Found an exact match. Collect all matches around this point.
            # This is not a strict binary search, but practical for this problem.
            i = mid
            while i >= 0 and keyword in books_sorted[i]['title'].lower():
                results.append(books_sorted[i])
                i -= 1
            i = mid + 1
            while i < len(books_sorted) and keyword in books_sorted[i]['title'].lower():
                results.append(books_sorted[i])
                i += 1
            break
        elif keyword < title:
            right = mid - 1
        else:
            left = mid + 1
    
    return results

def build_hash_index(books):
    """
    Builds a hash-based index for quick lookups by keywords.
    Time Complexity: O(n) for index creation.
    """
    index = {}
    for book in books:
        # Index by title words
        for word in book['title'].lower().split():
            index.setdefault(word, []).append(book)
        # Index by author words
        for word in book['author'].lower().split():
            index.setdefault(word, []).append(book)
    return index

def hash_search(index, keyword):
    """
    Performs a hash-based search using the pre-built index.
    Time Complexity: O(1) on average.
    """
    return index.get(keyword.lower(), [])

# --- Efficiency Comparison ---

def compare_searches(books, keyword):
    """
    Compares the execution time of different search algorithms.
    """
    print(f"--- Comparing Search Algorithms for '{keyword}' ---")

    # Linear Search
    start_time = time.perf_counter()
    linear_results = linear_search(books, keyword)
    end_time = time.perf_counter()
    linear_time = end_time - start_time
    print(f"Linear Search: {len(linear_results)} results found in {linear_time:.6f} seconds.")

    # Binary Search
    start_time = time.perf_counter()
    binary_results = binary_search_optimized(books, keyword)
    end_time = time.perf_counter()
    binary_time = end_time - start_time
    print(f"Binary Search: {len(binary_results)} results found in {binary_time:.6f} seconds.")

    # Hash-based Search (includes index building time for a fair comparison)
    start_time = time.perf_counter()
    index = build_hash_index(books)
    hash_results = hash_search(index, keyword)
    end_time = time.perf_counter()
    hash_time = end_time - start_time
    print(f"Hash Search:   {len(hash_results)} results found in {hash_time:.6f} seconds (including index build).")

    # Optional: Print results for verification
    # print("\nLinear Search Results:", linear_results)
    # print("Binary Search Results:", binary_results)
    # print("Hash Search Results:", hash_results)

# --- Main Execution ---
if __name__ == "__main__":
    
    # Static Input
    SEARCH_KEYWORD = "Douglas"
    
    # Run the comparison
    compare_searches(static_books, SEARCH_KEYWORD)