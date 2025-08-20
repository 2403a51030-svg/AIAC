import random

def collect_user_history():
    """
    Collects product history from the user.
    Returns a list of products the user has interacted with.
    """
    print("Welcome to the Product Recommender!")
    print("To help us recommend fairly, please enter products you've used or liked (type 'done' to finish):")
    user_history = []
    while True:
        product = input("Enter a product (or 'done' to finish): ").strip()
        if product.lower() == 'done':
            break
        if product:
            user_history.append(product)
    return user_history

def recommend_products(user_history, all_products, num_recommendations=3):
    """
    Recommend products based on user history.
    Ensures recommendations are fair and transparent.
    """
    # Transparency: Explain how recommendations are made
    print("\n[Transparency] Recommendations are based on your provided product history.")
    print("[Fairness] We avoid recommending the same product repeatedly and aim for diverse suggestions.\n")

    # Simple content-based filtering: recommend products similar to user's history
    # For demonstration, we recommend products not already in user history
    available_products = [p for p in all_products if p not in user_history]
    if not available_products:
        print("No new products to recommend. You've seen them all!")
        return []
    recommendations = random.sample(available_products, min(num_recommendations, len(available_products)))
    return recommendations

def main():
    # Example product catalog (in a real system, this would be much larger and more diverse)
    all_products = [
        "Wireless Headphones", "Smartphone", "Laptop", "Fitness Tracker", "E-reader",
        "Bluetooth Speaker", "Smartwatch", "Tablet", "Portable Charger", "Digital Camera"
    ]

    user_history = collect_user_history()
    if not user_history:
        print("No history provided. Please try again and enter at least one product.")
        return

    recommendations = recommend_products(user_history, all_products)
    if recommendations:
        print("We recommend you check out the following products:")
        for prod in recommendations:
            print(f"- {prod}")

    # Ethical guidelines reminder
    print("\n[Ethical Guidelines]")
    print("1. Transparency: You can ask how recommendations are made at any time.")
    print("2. Fairness: We strive to avoid bias and ensure all users receive diverse suggestions.")
    print("3. Privacy: Your product history is only used for this session and not stored.")

if __name__ == "__main__":
    main()
