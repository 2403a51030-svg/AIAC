"""
This script collects basic personally identifiable information (PII):
name, age, and email. It demonstrates safe handling practices.

Security and privacy guidance (read before using in production):
- Avoid logging or printing raw PII. Only display masked/summarized data.
- Obtain explicit user consent before storing or processing PII.
- Transmit PII only over secure channels (HTTPS/TLS).
- Store PII using a secure database with encryption at rest and in transit.
- Restrict access using least-privilege and audit access to PII.
- Comply with applicable regulations (e.g., GDPR/CCPA) and retention policies.
- Never hardcode secrets/keys in the source; use a secure secrets manager.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List


def prompt_for_non_empty_string(prompt_text: str) -> str:
    """Prompt until a non-empty string is provided."""
    while True:
        user_input = input(prompt_text).strip()
        if user_input:
            return user_input
        print("Input cannot be empty. Please try again.")


def prompt_for_age(prompt_text: str = "Enter your age: ") -> int:
    """Prompt until a valid age (positive integer) is provided."""
    while True:
        raw_value = input(prompt_text).strip()
        if not raw_value.isdigit():
            print("Age must be a positive whole number. Please try again.")
            continue
        age_value = int(raw_value)
        if age_value <= 0 or age_value > 120:
            print("Please enter a realistic age between 1 and 120.")
            continue
        return age_value


def is_valid_email(email: str) -> bool:
    """Basic email format validation.

    Note: This is a simple regex-based validation suitable for basic checks.
    For production, consider more robust validation strategies.
    """
    email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    return bool(email_pattern.match(email))


def prompt_for_email(prompt_text: str = "Enter your email: ") -> str:
    """Prompt until a string that looks like an email address is provided."""
    while True:
        email_value = input(prompt_text).strip()
        if is_valid_email(email_value):
            return email_value
        print("That doesn't look like a valid email. Please try again.")


def mask_name_for_display(name: str) -> str:
    """Mask a name for safe display (avoid revealing full PII in console)."""
    trimmed = name.strip()
    if len(trimmed) <= 2:
        return trimmed[0] + "*" * (len(trimmed) - 1)
    return f"{trimmed[0]}***{trimmed[-1]}"


def mask_email_for_display(email: str) -> str:
    """Mask an email for safe display (e.g., a*****@example.com)."""
    if "@" not in email:
        return "***"
    local_part, domain_part = email.split("@", 1)
    if not local_part:
        return "***@" + domain_part
    return local_part[0] + "*" * max(1, len(local_part) - 1) + "@" + domain_part


def build_storable_record(raw_record: Dict[str, str | int]) -> Dict[str, str | int]:
    """Create a storable record that avoids raw PII.

    Anonymization techniques illustrated here:
    - Remove direct identifiers (do not store raw email/name here).
    - Mask sensitive fields for display/storage (e.g., j***@gmail.com).
    - Consider pseudonymization (e.g., salted hashes) or encryption for reversibility.

    For production systems, prefer ENCRYPTION of sensitive fields at rest and in transit.
    """
    return {
        # Store masked values instead of raw PII.
        "name_masked": mask_name_for_display(str(raw_record.get("name", ""))),
        "age": int(raw_record.get("age", 0)),
        "email_masked": mask_email_for_display(str(raw_record.get("email", ""))),
        # Example of removing direct identifiers: do not include raw name/email fields.
    }


def securely_save_user(user_record: Dict[str, str | int]) -> None:
    """Persist a protected version of the data to local JSON.

    This demo writes ONLY masked fields to `users_masked.json` to avoid storing raw PII.
    Notes for production:
    - Use a secure database and encrypt sensitive fields (with a KMS-managed key).
    - Apply least privilege access and maintain audit logs.
    - Consider data minimization: store only what you need, for as long as needed.
    """
    output_file = "users_masked.json"

    storable: Dict[str, str | int] = build_storable_record(user_record)

    existing: List[Dict[str, str | int]] = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            # If the file is corrupted or not JSON, start fresh rather than exposing PII via logs.
            existing = []

    existing.append(storable)

    # Write back to disk with indentation for readability.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def main() -> None:
    # Collect PII with basic validation. Avoid printing raw values.
    full_name = prompt_for_non_empty_string("Enter your full name: ")
    age_years = prompt_for_age()
    email_address = prompt_for_email()

    user_record: Dict[str, str | int] = {
        "name": full_name,
        "age": age_years,
        "email": email_address,
    }

    # Display only masked data to reduce the risk of exposing PII.
    print("\nThank you. Here's a masked summary of what you entered:")
    print(f"- Name:  {mask_name_for_display(full_name)}")
    print(f"- Age:   {age_years}")  # Age alone is less sensitive but still PII
    print(f"- Email: {mask_email_for_display(email_address)}")

    # Obtain consent before saving/storing the data.
    consent = input("\nDo you consent to store this information securely? (y/N): ").strip().lower()
    if consent == "y":
        securely_save_user(user_record)
        print("Your data has been recorded securely (demo placeholder).")
    else:
        print("Your data was not stored.")


if __name__ == "__main__":
    main()



