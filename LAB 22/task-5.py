def loan_approval_system():
    print("Welcome to the Loan Approval System") 
    # User inputs
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    income = float(input("Enter your annual income: "))
    loan_amount = float(input("Enter the loan amount you wish to apply for: "))
    credit_score = int(input("Enter your credit score (0-850): "))
    # Approval criteria
    if age < 18:
        print("Loan not approved: You must be at least 18 years old.")
    elif income < 30000:
        print("Loan not approved: Your annual income must be at least $30,000.")
    elif loan_amount > income * 5:
        print("Loan not approved: You cannot borrow more than 5 times your annual income.")
    elif credit_score < 600:
        print("Loan not approved: Your credit score must be at least 600.")
    else:
        print(f"Congratulations {name}, your loan of ${loan_amount} is approved!")

if __name__ == "__main__":
    loan_approval_system()