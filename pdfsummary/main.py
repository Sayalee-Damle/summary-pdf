import streamlit as st
import extractkeyword as ek
import vector_embedding as ve
from streamlit_chat import message
from pathlib import Path

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def save_in_memory(prompt):
    st.session_state.messages.append({"role": "assistant", "content": prompt})
    
def use_prompt():
    val =  st.chat_input("Type here")
    with st.chat_message("user"):
        st.write(val)
    st.session_state.messages.append({"role": "user", "content": val})
    return val


st.header("Read your PDF and Get it Summarized! Also ask questions and get the Summary Translated if you want!")
path_pdf = "Enter your pdf file path "
with st.chat_message("assistant"):
    st.write(path_pdf)

if prompt:= st.chat_input("Type here"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Generating response"):
        db, pages = ve.init_vector_store(Path(prompt))
        summary = 'summary' #ek.summary_llm(pages)
        with st.chat_message("assistant"):
            st.write(summary)
        st.session_state.messages.append({"role": "assistant", "content": summary})
        #display summary
    translate = "Do you want to translate the summary (Yes/No)?"
    with st.chat_message("assistant"):
        st.markdown(translate)
    
    
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add user message to chat history
    #st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Add assistant response to chat history
        #st.session_state.messages.append({"role": "assistant", "content": response})

