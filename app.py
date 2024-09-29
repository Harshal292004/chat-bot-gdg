import os
from dotenv import load_dotenv
import streamlit as st
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import re
from langdetect import detect

# Load environment variables
load_dotenv()

# Set up the API key
api_key = os.getenv('HUGGINGFACE_API_KEY')
if not api_key:
    raise ValueError("No API key found. Please set the HUGGINGFACE_API_KEY environment variable.")

# Set up the model
model = "HuggingFaceH4/starchat-beta"
llm = HuggingFaceHub(
    repo_id=model,
    huggingfacehub_api_token="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",
    model_kwargs={
        "min_length": 30,
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_k": 50,
        "top_p": 0.95,
        "eos_token_id": 49155
    }
)
# Set up the prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="Human: {user_input}\nAI: "
)

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

def extract_english(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    english_sentences = []
    for sentence in sentences:
        try:
            if detect(sentence) == 'en':
                english_sentences.append(sentence)
            else:
                # Stop when we encounter a non-English sentence
                break
        except:
            # If language detection fails, assume it's not English
            break
    
    return ' '.join(english_sentences).strip()



def generate_response(user_input):
    try:
        llm_reply = llm_chain.run(user_input)
        # Split the reply at "AI:" and take the last part
        reply = llm_reply.split("AI:")[-1].strip()
        english_reply = extract_english(reply)
        return english_reply
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("AI Assistant")

user_input = st.text_input("You:", "")
if st.button("Send"):
    response = generate_response(user_input)
    st.text_area("Chatbot:", response, height=200)