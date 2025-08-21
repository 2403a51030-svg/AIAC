from typing import Dict, Tuple, List
class ShoppingList:
    def __init__(self) -> None:
        self.items: Dict[str, Dict[str, float | int]] = {}

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        if not name:
            print("Item name cannot be empty.")
            return
        if price <= 0:
            print("Price must be positive.")
            return
        if quantity <= 0:
            print("Quantity must be positive.")
            return
        if name in self.items:
            self.items[name]["quantity"] += quantity
            self.items[name]["price"] = float(price)
        else:
            self.items[name] = {"price": float(price), "quantity": int(quantity)}
        print(f"Added {quantity} x {name} @ {price:.2f}")

    def remove_item(self, name: str, quantity: int | None = None) -> None:
        if name not in self.items:
            print("Item not found.")
            return
        if quantity is None or quantity >= self.items[name]["quantity"]:
            del self.items[name]
            print(f"Removed all of {name}.")
            return
        if quantity <= 0:
            print("Quantity must be positive.")
            return
        self.items[name]["quantity"] -= quantity
        print(f"Removed {quantity} of {name}.")

    def total_bill(self) -> float:
        return sum(details["price"] * details["quantity"] for details in self.items.values())

    def list_items(self) -> List[Tuple[str, float, int]]:
        return [(name, data["price"], data["quantity"]) for name, data in self.items.items()]

    def is_empty(self) -> bool:
        return len(self.items) == 0


def print_menu() -> None:
    print("\nShopping List Menu")
    print("1) Add item")
    print("2) Remove item")
    print("3) Show items")
    print("4) Show total bill")
    print("0) Exit")


def main() -> None:
    sl = ShoppingList()
    while True:
        print_menu()
        choice = input("Choose an option: ").strip()
        if choice == "1":
            name = input("Item name: ").strip()
            try:
                price = float(input("Price: ").strip())
                qty = int(input("Quantity: ").strip())
            except ValueError:
                print("Invalid price or quantity.")
                continue
            sl.add_item(name, price, qty)
        elif choice == "2":
            name = input("Item name to remove: ").strip()
            qty_text = input("Quantity to remove (leave blank to remove all): ").strip()
            qty = None
            if qty_text:
                try:
                    qty = int(qty_text)
                except ValueError:
                    print("Invalid quantity.")
                    continue
            sl.remove_item(name, qty)
        elif choice == "3":
            if sl.is_empty():
                print("No items in the list.")
            else:
                print("\nCurrent items:")
                for name, price, qty in sl.list_items():
                    print(f"- {name}: {qty} @ {price:.2f} each = {qty * price:.2f}")
        elif choice == "4":
            print(f"Total bill: {sl.total_bill():.2f}")
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()


