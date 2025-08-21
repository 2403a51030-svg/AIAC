# Define student names and scores using a dictionary
students = {
    "Alice": 92,
    "Bob": 68,
    "Charlie": 77,
    "Dana": 84,
    "Eve": 73,
}

threshold = 75

print("Students scoring above 75:")

# Use a while loop to iterate through dictionary items
items = list(students.items())
index = 0

while index < len(items):
    name, score = items[index]
    if score > threshold:
        print(f"{name} - {score}")
    index += 1


