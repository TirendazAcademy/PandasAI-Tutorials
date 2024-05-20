import streamlit as st 
import pandas as pd
from langchain_openai import ChatOpenAI
from pandasai import SmartDataframe

st.set_page_config(page_title="Talk to Your Data")
st.title("Talk to Your Data 🐼")

st.sidebar.image("data/pandasai.png", use_column_width=True)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

openai_api_key = st.sidebar.text_input("OpenAI API Key", type = "password", 
                                       disabled=not (uploaded_file)) 

if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠️")

prompt = st.text_input("Enter your prompt")

def generate_response(csv_file, prompt):
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", openai_api_key=openai_api_key, 
                     temperature=0)
    df = SmartDataframe(data, config={"llm":llm})
    response = df.chat(prompt)
    return st.write(response)

if st.button("Ask your data"):
    if openai_api_key.startswith("sk-") and (uploaded_file is not None):
        if prompt:
            with st.spinner("Generating response..."):
                generate_response(uploaded_file, prompt)