import random
import string
import time

# Generate random student records
def generate_students(n):
    students = []
    for i in range(n):
        name = ''.join(random.choices(string.ascii_uppercase, k=7))
        roll_no = f"R{1000+i}"
        cgpa = round(random.uniform(5.0, 10.0), 2)
        students.append({'Name': name, 'Roll No': roll_no, 'CGPA': cgpa})
    return students

# Quick Sort implementation
def quick_sort(arr, key):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2][key]
    left = [x for x in arr if x[key] > pivot]
    middle = [x for x in arr if x[key] == pivot]
    right = [x for x in arr if x[key] < pivot]
    return quick_sort(left, key) + middle + quick_sort(right, key)

# Merge Sort implementation
def merge_sort(arr, key):
    if len(arr) <= 1:
        return arr
    mid = len(arr)//2
    left = merge_sort(arr[:mid], key)
    right = merge_sort(arr[mid:], key)
    return merge(left, right, key)

def merge(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][key] > right[j][key]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Function to output top 10 students
def top_10_students(sorted_students):
    print("Top 10 Students by CGPA:")
    for i, student in enumerate(sorted_students[:10], 1):
        print(f"{i}. Name: {student['Name']}, Roll No: {student['Roll No']}, CGPA: {student['CGPA']}")

if __name__ == "__main__":
    n = 10000  # Large dataset
    students = generate_students(n)

    # Quick Sort
    start = time.time()
    quick_sorted = quick_sort(students, 'CGPA')
    quick_time = time.time() - start
    print(f"Quick Sort Time: {quick_time:.4f} seconds")

    # Merge Sort
    start = time.time()
    merge_sorted = merge_sort(students, 'CGPA')
    merge_time = time.time() - start
    print(f"Merge Sort Time: {merge_time:.4f} seconds")

    # Output top 10 students
    top_10_students(quick_sorted)