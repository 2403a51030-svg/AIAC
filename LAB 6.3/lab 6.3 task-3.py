class BankAccount:
    def __init__(self, account_holder, balance=0.0):
        self.account_holder = account_holder
        self.balance = float(balance)

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited: {amount}")
        else:
            print("Amount must be positive")

    def withdraw(self, amount):
        if amount <= 0:
            print("Amount must be positive")
            return
        if amount > self.balance:
            print("Insufficient balance")
        else:
            self.balance -= amount
            print(f"Withdrawn: {amount}")

    def get_balance(self):
        return self.balance


if __name__ == "__main__":
    name = input("Enter account holder name: ")
    try:
        starting = float(input("Enter starting balance (default 0): ") or 0)
    except ValueError:
        starting = 0.0

    account = BankAccount(name, starting)
    print(f"\nCreated account for {account.account_holder} with balance {account.get_balance():.2f}")

    # Deposit
    try:
        dep = float(input("Enter deposit amount (or 0 to skip): ") or 0)
    except ValueError:
        dep = 0.0
    if dep:
        account.deposit(dep)

    # Withdraw
    try:
        wd = float(input("Enter withdrawal amount (or 0 to skip): ") or 0)
    except ValueError:
        wd = 0.0
    if wd:
        account.withdraw(wd)

    print(f"\nFinal Balance: {account.get_balance():.2f}")


