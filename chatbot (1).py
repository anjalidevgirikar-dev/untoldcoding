# chatbot.py

# Dictionary of FAQs
faqs = {
    "what is your name": "I am a FAQ chatbot created to answer your questions.",
    "how can i contact support": "You can contact support via email: support@example.com.",
    "what are your working hours": "Our working hours are 9 AM to 6 PM, Monday to Friday.",
    "how do i reset my password": "Click on 'Forgot Password' at the login page and follow the instructions.",
    "thank you": "You're welcome! 😊"
}

def chatbot():
    print("Welcome to the FAQ Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").lower()
        if user_input == "exit":
            print("Chatbot: Goodbye!")
            break
        response = faqs.get(user_input, "Sorry, I don't know the answer to that. Please contact support.")
        print("Chatbot:", response)

if __name__ == "__main__":
    chatbot()
