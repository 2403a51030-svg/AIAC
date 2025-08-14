import re

def extract_emails(text):
    # Regular expression pattern for matching email addresses
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    return re.findall(email_pattern, text)

# Example usage
if __name__ == "__main__":
    sample_text = """
    Please contact us at support@example.com or sales-info@company.co.uk.
    You can also reach out to admin123@domain.org for more information.
    """
    emails = extract_emails(sample_text)
    print("Extracted email addresses:")
    for email in emails:
        print(email)