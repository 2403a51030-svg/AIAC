from typing import Optional
class Student:
    """Represents an SRU student with core attributes and operations."""

    def __init__(self, name: str, roll_no: str, hostel_status: bool, fee_paid: int = 0) -> None:
        self.name = name
        self.roll_no = roll_no
        self.hostel_status = hostel_status
        self.fee_paid = fee_paid

    def update_fee(self, amount: int) -> None:
        """Increase the student's fee paid by a positive amount."""
        if amount < 0:
            raise ValueError("Amount must be non-negative")
        self.fee_paid += amount

    def set_hostel_status(self, status: bool) -> None:
        """Set hostel status (True for hosteller, False for day scholar)."""
        self.hostel_status = status

    def display_details(self) -> str:
        """Return a formatted string of the student's details."""
        hostel_text = "Hosteller" if self.hostel_status else "Day Scholar"
        return (
            f"Name: {self.name}\n"
            f"Roll No: {self.roll_no}\n"
            f"Hostel Status: {hostel_text}\n"
            f"Fee Paid: {self.fee_paid}"
        )


def _parse_bool(text: str) -> bool:
    text_l = text.strip().lower()
    return text_l in {"y", "yes", "true", "1"}


def _input_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This field is required. Please enter a value.")


def _input_int(prompt: str, default: Optional[int] = None) -> int:
    while True:
        text = input(prompt).strip()
        if not text and default is not None:
            return default
        try:
            return int(text)
        except ValueError:
            print("Please enter a valid integer.")


if __name__ == "__main__":
    print("Enter SRU Student details")
    name = _input_non_empty("Name: ")
    roll_no = _input_non_empty("Roll No: ")
    hostel_status = _parse_bool(input("Hosteller? (y/n): ").strip())
    fee_paid = _input_int("Initial fee paid (integer, default 0): ", default=0)

    student = Student(name=name, roll_no=roll_no, hostel_status=hostel_status, fee_paid=fee_paid)

    print("\nCurrent Details:")
    print(student.display_details())

    if _parse_bool(input("\nDo you want to update fee? (y/n): ").strip()):
        add_amt = _input_int("Enter amount to add to fee (integer): ")
        try:
            student.update_fee(add_amt)
        except ValueError:
            print("Amount must be non-negative. Skipping fee update.")

    if _parse_bool(input("Change hostel status? (y/n): ").strip()):
        student.set_hostel_status(_parse_bool(input("Set hosteller? (y/n): ").strip()))

    print("\nFinal Details:")
    print(student.display_details())
