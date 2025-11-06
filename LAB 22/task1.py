import json
import uuid
import datetime
import os

# task1.py
# Simple CLI Customer Support Chatbot
# Usage: python task1.py


KB = {
    "billing": {
        "keywords": ["bill", "billing", "charge", "payment", "invoice", "refund"],
        "response": "It looks like a billing question. Common steps:\n"
                    "1. Check your last invoice in your account.\n"
                    "2. Confirm payment method and date.\n"
                    "3. If incorrect charge, request a refund using option 'escalate'."
    },
    "technical": {
        "keywords": ["error", "bug", "crash", "not working", "issue", "problem", "fail"],
        "response": "It sounds technical. Try:\n"
                    "1. Restart the app/device.\n"
                    "2. Update to the latest version.\n"
                    "3. Reproduce the error and note exact messages."
    },
    "account": {
        "keywords": ["login", "password", "signup", "account", "username", "verify", "verification"],
        "response": "Account-related issue. Steps:\n"
                    "1. Use 'Forgot Password' to reset.\n"
                    "2. Ensure email is verified.\n"
                    "3. If locked out, choose 'escalate' for manual review."
    },
    "shipping": {
        "keywords": ["shipping", "delivery", "track", "package", "shipment", "delay"],
        "response": "Shipping query. Suggestions:\n"
                    "1. Check tracking number from order page.\n"
                    "2. Allow transit time; contact carrier for delays.\n"
                    "3. Escalate if package lost."
    }
}

TICKETS_FILE = "tickets.json"


def collect_user_details():
    print("Please enter your details (leave blank to skip):")
    name = input("Name: ").strip()
    email = input("Email: ").strip()
    phone = input("Phone: ").strip()
    return {"name": name, "email": email, "phone": phone}


def find_best_category(message):
    msg = message.lower()
    scores = {}
    for cat, info in KB.items():
        for kw in info["keywords"]:
            if kw in msg:
                scores[cat] = scores.get(cat, 0) + 1
    if not scores:
        return None
    # choose category with highest count
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best


def suggest_response(category):
    if category and category in KB:
        return KB[category]["response"]
    return ("I couldn't identify the exact issue. Please describe the problem in more detail, "
            "or type 'options' to see available actions.")


def save_ticket(user, initial_message, category, convo):
    ticket = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "user": user,
        "category": category or "uncategorized",
        "initial_message": initial_message,
        "conversation": convo
    }
    # load existing
    if os.path.exists(TICKETS_FILE):
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    else:
        data = []
    data.append(ticket)
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return ticket["id"]


def print_options():
    print("\nOptions:")
    print("  help     - Repeat this help")
    print("  escalate - Create a support ticket and escalate to human agent")
    print("  save     - Save current conversation to a ticket file (local)")
    print("  quit     - Exit chat\n")


def chat_loop(user):
    print("\nStart describing your issue. Type 'help' for options.")
    convo = []
    initial_message = None
    while True:
        try:
            msg = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye.")
            break
        if not msg:
            continue
        if initial_message is None:
            initial_message = msg
        convo.append({"from": "user", "text": msg, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        cmd = msg.lower()
        if cmd in ("quit", "exit"):
            print("Goodbye.")
            break
        if cmd == "help":
            print_options()
            continue
        if cmd == "options":
            print_options()
            continue
        if cmd == "escalate":
            category = find_best_category(initial_message or "")
            ticket_id = save_ticket(user, initial_message or "", category, convo)
            print(f"Your issue has been escalated. Ticket ID: {ticket_id}")
            break
        if cmd == "save":
            category = find_best_category(initial_message or "")
            ticket_id = save_ticket(user, initial_message or "", category, convo)
            print(f"Conversation saved locally as ticket {ticket_id}.")
            continue

        # generate assistant reply (simple rule-based)
        category = find_best_category(msg)=
        response = suggest_response(category)
        convo.append({"from": "bot", "text": response, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        print("\nBot:", response)
        print("\nIf this doesn't help, type 'escalate' to create a support ticket or continue describing the issue.")


def main():
    print("Customer Support Chatbot\n------------------------")
    user = collect_user_details()
    print("\nThank you. Your details:", {k: v for k, v in user.items() if v})
    chat_loop(user)


if __name__ == "__main__":
    main()