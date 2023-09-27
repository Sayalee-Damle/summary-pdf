import streamlit as st
import extractkeyword as extractk
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

ques1 = "Enter your pdf file path "

with st.chat_message("assistant"): 
    path_pdf = st.text_input(ques1)


if path_pdf:
    with st.chat_message("user"): 
        st.write(path_pdf)
    st.session_state.messages.append({"role": "user", "content": path_pdf})
    db, pages = ve.init_vector_store(Path(path_pdf))
    summary = extractk.summary_llm(pages)
    with st.chat_message("assistant"):
        st.write(summary)
    st.session_state.messages.append({"role": "assistant", "content": summary})

    ques2 = "Do you want to translate the summary (Yes/No)?"
    with st.chat_message("assistant"):
        answer = st.text_input(ques2)
    if answer:
        with st.chat_message("user"):
            st.write(answer)
        st.session_state.messages.append({"role": "user", "content": answer})

        if answer.lower() == "yes":
            ques3 = "Which language?"
            with st.chat_message("assistant"):
                lang = st.text_input(ques3)
            #prompt = None
            if lang:
                with st.chat_message("user"):
                    st.write(lang)
                st.session_state.messages.append({"role": "user", "content": lang})
                translation = extractk.translation(summary, lang)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(translation)
                st.session_state.messages.append({"role": "user", "content": translation})
