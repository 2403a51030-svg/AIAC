import argparse
import sys


def compute_grade(marks: float) -> str:

    if marks >= 90:
        return "A"
    if marks >= 75:
        return "B"
    if marks >= 60:
        return "C"
    return "Fail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display student details and calculate grade from marks.",
    )
    parser.add_argument("--name", type=str, help="Student name")
    parser.add_argument("--roll", type=int, help="Student roll number")
    parser.add_argument("--marks", type=float, help="Marks (0-100)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    name = args.name
    roll_number = args.roll
    marks = args.marks

    if not name:
        name = input("Enter student name: ").strip()
    if roll_number is None:
        try:
            roll_number = int(input("Enter roll number: ").strip())
        except ValueError:
            print("Invalid roll number. Please enter an integer.")
            2sys.exit(1)
    if marks is None:
        try:
            marks = float(input("Enter marks (0-100): ").strip())
        except ValueError:
            print("Invalid marks. Please enter a number.")
            sys.exit(1)

    if marks < 0 or marks > 100:
        print("Marks must be between 0 and 100.")
        sys.exit(1)

    grade = compute_grade(marks)

    print("\nStudent Details")
    print("----------------")
    print(f"Name      : {name}")
    print(f"Roll No.  : {roll_number}")
    print(f"Marks     : {marks}")
    print(f"Grade     : {grade}")


if __name__ == "__main__":
    main()


