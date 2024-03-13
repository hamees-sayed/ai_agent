import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine import PandasQueryEngine
from prompt import new_prompt, instruction_str

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def generate_response(file_path, query, instruction_str=instruction_str, new_prompt=new_prompt, temperature=0.5):
    df = pd.read_csv(file_path)
    llm = Gemini(model="gemini-pro", temperature=temperature)

    query_engine = PandasQueryEngine(df=df, llm=llm, instruction_str=instruction_str)
    query_engine.update_prompts({"pandas_prompt": new_prompt})

    return query_engine.query(query)


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
            st.chat_message("user").write(query)
            st.session_state.messages.append({"role": "user", "content": query})
            print(csv.name, query)
            response = generate_response(csv, query, temperature)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            

if __name__ == "__main__":
    main()
    