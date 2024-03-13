import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine import PandasQueryEngine
from pandasai import SmartDataframe
from pandasai.llm.google_palm import GooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
#from prompt import new_prompt, instruction_str

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def png_exists(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            return True
    return False

def delete_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)


def generate_response(file_path, query, temperature=0.5):
    df = pd.read_csv(file_path)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)

    pdi = SmartDataframe(df, config={"llm": llm})
    
    return pdi.chat(query)


def main():
    with st.sidebar:
        st.title('DocAssist 💬')
        st.header("1. Upload PDF")
        csv = st.file_uploader("**Upload your CSV File**", type='csv')
        temperature = st.slider(
            "Select the creativity temperature for the AI",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
    
    query = st.chat_input("What is up?")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if query:
        if csv is not None:
            delete_png("./exports/charts")
            st.chat_message("user").write(query)
            st.session_state.messages.append({"role": "user", "content": query})
            response = generate_response(csv, query, temperature)
            with st.chat_message("assistant"):
                if png_exists("./exports/charts"):
                    st.image(response)
                else:
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            

if __name__ == "__main__":
    main()
    