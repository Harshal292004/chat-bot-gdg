# AI Assistant Chatbot

This project is an AI Assistant chatbot built using Streamlit and the HuggingFace Starchat-beta model. It allows users to interact with an AI assistant by entering their HuggingFace API token and sending messages.

## Live Demo

You can try out the live demo of this chatbot at:
[https://chat-bot-gdg.streamlit.app/](https://chat-bot-gdg.streamlit.app/)

## Local Setup

To run this project locally, follow these steps:

1. Clone the repository:

```bash 
git clone https://github.com/yourusername/ai-assistant-chatbot.git
```
2. Install the required packages:

```bash 
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
    streamlit run app.py
```
or
```bash 
    python -m streamlit run app.py
```

5. Open your web browser and go to `http://localhost:8501` to use the chatbot.

## Usage

1. Enter your HuggingFace API token in the provided input field.
2. Type your message in the "You:" input box.
3. Click the "Send" button to get a response from the AI assistant.

## Requirements

The main requirements for this project are:

- streamlit
- langchain
- langdetect

For a complete list of dependencies, see the `requirements.txt` file.

## Note

Make sure you have a valid HuggingFace API token to use this chatbot. You can obtain one by signing up at [HuggingFace](https://huggingface.co/).
