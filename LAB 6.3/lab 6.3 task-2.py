import argparse
import sys
from typing import Iterable, List, Tuple


def is_even(number: int) -> bool:

    if number % 2 == 0:
        return True
    else:
        return False


def compute_even_squares(numbers: Iterable[int]) -> List[Tuple[int, int]]:
    even_squares: List[Tuple[int, int]] = []
    for value in numbers:
        if is_even(value):
            even_squares.append((value, value * value))
    return even_squares


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate squares of even numbers from provided integers.",
    )
    parser.add_argument(
        "numbers",
        nargs="*",
        type=int,
        help="Integers to process. If omitted, you'll be prompted to enter them.",
    )
    return parser.parse_args()


def read_numbers_interactively() -> List[int]:
    raw = input("Enter integers separated by spaces: ").strip()
    if not raw:
        return []
    try:
        return [int(part) for part in raw.split()]
    except ValueError:
        print("Invalid input. Please enter only integers separated by spaces.")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    numbers: List[int] = args.numbers if args.numbers else read_numbers_interactively()

    results = compute_even_squares(numbers)

    if not results:
        print("No even numbers found.")
        return

    print("Even numbers and their squares:")
    for value, square in results:
        print(f"{value}^2 = {square}")


if __name__ == "__main__":
    main()


