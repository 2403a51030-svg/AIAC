from typing import Union

Number = Union[int, float]


def add(a: Number, b: Number) -> Number:
    """Add two numbers.

    Parameters
    ----------
    a : int or float
        First addend.
    b : int or float
        Second addend.

    Returns
    -------
    int or float
        The sum of ``a`` and ``b``.

    Examples
    --------
    >>> add(2, 3)
    5
    >>> add(2.5, 0.5)
    3.0
    """
    return a + b


def subtract(a: Number, b: Number) -> Number:
    """Subtract one number from another.

    Parameters
    ----------
    a : int or float
        Minuend.
    b : int or float
        Subtrahend.

    Returns
    -------
    int or float
        The difference ``a - b``.

    Examples
    --------
    >>> subtract(5, 3)
    2
    >>> subtract(1.5, 2)
    -0.5
    """
    return a - b


def multiply(a: Number, b: Number) -> Number:
    """Multiply two numbers.

    Parameters
    ----------
    a : int or float
        First factor.
    b : int or float
        Second factor.

    Returns
    -------
    int or float
        The product of ``a`` and ``b``.

    Examples
    --------
    >>> multiply(4, 3)
    12
    >>> multiply(2.5, 2)
    5.0
    """
    return a * b


def divide(a: Number, b: Number) -> float:
    """Divide one number by another.

    Parameters
    ----------
    a : int or float
        Dividend.
    b : int or float
        Divisor. Must be non-zero.

    Returns
    -------
    float
        The quotient ``a / b`` as a float.

    Raises
    ------
    ZeroDivisionError
        If ``b`` is zero.

    Examples
    --------
    >>> divide(10, 2)
    5.0
    >>> divide(3, 2)
    1.5
    """
    if b == 0:
        raise ZeroDivisionError("Division by zero is not allowed")
    return a / b


if __name__ == "__main__":
    print("NumPy-style Calculator")

    def _input_number(prompt: str) -> Number:
        while True:
            text = input(prompt).strip()
            try:
                # Try int first for cleaner integers; fallback to float
                if "." in text or "e" in text.lower():
                    return float(text)
                return int(text)
            except ValueError:
                print("Please enter a valid number (e.g., 10 or 3.14).")

    a = _input_number("Enter first number: ")
    b = _input_number("Enter second number: ")

    ops = {"+": add, "-": subtract, "*": multiply, "/": divide}

    while True:
        op = input("Choose operation (+, -, *, /): ").strip()
        if op in ops:
            try:
                result = ops[op](a, b)
            except ZeroDivisionError as e:
                print(str(e))
                continue
            print(f"Result: {a} {op} {b} = {result}")
            break
        else:
            print("Invalid operation. Please choose one of +, -, *, /")
