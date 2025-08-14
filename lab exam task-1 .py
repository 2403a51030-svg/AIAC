
def convert_currency(amount, from_currency, to_currency, exchange_rates):
    """
    Converts an amount from one currency to another using exchange rates.

    :param amount: The amount of money to convert.
    :param from_currency: The currency code of the original currency.
    :param to_currency: The currency code to convert to.
    :param exchange_rates: Dictionary with currency codes as keys and rates as values (relative to a base currency).
    :return: Converted amount in the target currency.
    """
    if from_currency not in exchange_rates or to_currency not in exchange_rates:
        raise ValueError("Currency code not found in exchange rates.")
    # Convert amount to base currency first, then to target currency
    base_amount = amount / exchange_rates[from_currency]
    converted_amount = base_amount * exchange_rates[to_currency]
    return converted_amount

# Example usage:
exchange_rates = {
    'USD': 1.0,      # base currency
    'EUR': 0.92,
    'INR': 83.2,
    'GBP': 0.78
}

# Convert 100 USD to INR
result = convert_currency(100, 'USD', 'INR', exchange_rates)
print(result)  # Output: 8320.0