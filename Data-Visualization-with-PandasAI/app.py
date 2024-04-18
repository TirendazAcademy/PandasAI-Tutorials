from dotenv import load_dotenv
import streamlit as st
import seaborn as sns
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataframe

load_dotenv()

st.title("Data visualization with PandasAI")

data = sns.load_dataset("penguins")
st.write(data.head(3))

model = ChatAnthropic(model="claude-3-haiku-20240307")
df = SmartDataframe(data, config={"llm": model})

prompt = st.text_area("Enter your prompt:")

if st.button("Generate:"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write(df.chat(prompt))