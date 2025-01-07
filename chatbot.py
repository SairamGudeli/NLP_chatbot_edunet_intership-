import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

# Function for custom CSS
def add_custom_css():
    st.markdown("""
        <style>
            /* General page background color */
            body {
                background-color: #f9f9f9;
                font-family: Arial, sans-serif;
            }

            /* Chat area styling */
            .stTextInput>div>div>input {
                background-color: #333333;
                color: #ffffff;
                font-size: 16px;
                border-radius: 15px;
                padding: 10px;
                border: 1px solid #ccc;
            }

            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 10px;
                width: 100%;
                cursor: pointer;
            }

            /* Hover effect on button */
            .stButton>button:hover {
                background-color: #45a049;
            }

            .chatbox {
                padding: 15px;
                background-color: #ffffff;
                border-radius: 10px;
                max-width: 80%;
                margin: 10px 0;
                border: 1px solid #ddd;
            }

            .chatbot {
                color: #4CAF50;
                font-weight: bold;
            }

            .user {
                color: #007bff;
            }

            .stTextArea>div>textarea {
                font-size: 16px;
            }

            /* Sidebar styling */
            .stSidebar {
                background-color: #FF7F50;
            }
        </style>
        """, unsafe_allow_html=True)

def main():
    global counter
    st.title("Chatbot with Natural Language Processing 🤖")

    # Add custom CSS for better design
    add_custom_css()

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation. 😊")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day! 👋")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History 📜")
        st.write("Here is the log of previous conversations. 📝")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.markdown(f"**User:** {row[0]}")
                st.markdown(f"**Chatbot:** {row[1]}")
                st.markdown(f"**Timestamp:** {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) and Logistic Regression, which allows it to predict the intent and generate relevant responses.")
        st.subheader("Project Overview:")
        st.write("""
        - **NLP** and **Logistic Regression** are used to train the chatbot.
        - **Streamlit** provides the framework to build the interactive web interface.
        """)
        st.subheader("Dataset:")
        st.write("""
        The dataset consists of labeled **intents** and **patterns** that help the chatbot understand user queries.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("""
        The interface allows users to chat with the chatbot and receive responses in real-time.
        """)

        st.subheader("Conclusion:")
        st.write("The chatbot can be extended by adding more intents, using sophisticated NLP techniques, or integrating deep learning algorithms.")

if __name__ == '__main__':
    main()
