import os
from dotenv import load_dotenv
import streamlit as st
from langchain import HuggingFaceHub, LLMChain, PromptTemplate

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
    huggingfacehub_api_token=api_key,
    model_kwargs={
        "min_length": 30,
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 0.95,
        "eos_token_id": 49155
    }
)
# Set up the prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="Human: {user_input}\n respond in a helpful and friendly manner AI:"
)

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

def generate_response(user_input):
    try:
        llm_reply = llm_chain.run(user_input)
        reply = llm_reply.partition('<|end|>')[0]
        return reply
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("AI Assistant")

user_input = st.text_input("You:", "")
if st.button("Send"):
    response = generate_response(user_input)
    st.text_area("Chatbot:", response, height=200)