from langchain.schema import BaseOutputParser
from config import Config
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredPDFLoader
import vector_embedding as ve
import templates as t
import user_input
from pathlib import Path
from prompt_toolkit import prompt


class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")
        



path_pdf = user_input.get_path_pdf()
db, pages = ve.init_vector_store(path_pdf)
    
#to find summary
def summary_llm(pages):
    system_message_prompt = SystemMessagePromptTemplate.from_template(t.summary_template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm = Config.llm, prompt = chat_prompt)
    summary = chain.run({'text': pages})
    return summary

def translation(summary, language):
    system_message_prompt_3 = SystemMessagePromptTemplate.from_template(t.system_translate_template)
    human_message_prompt_3 = HumanMessagePromptTemplate.from_template(t.translate_template)
    chat_prompt_3 = ChatPromptTemplate.from_messages([system_message_prompt_3, human_message_prompt_3])
    chain3 = LLMChain(llm = Config.llm, prompt = chat_prompt_3)
    return chain3.run({'text': summary, 'language': language})
   
#print(pages[0].page_content)

#template for question answer
def question_llm(query, db):
    system_message_prompt_2 = SystemMessagePromptTemplate.from_template(t.system_template)
    human_message_prompt_2 = HumanMessagePromptTemplate.from_template(t.template2)
    chat_prompt_2 = ChatPromptTemplate.from_messages([system_message_prompt_2, human_message_prompt_2])
    docs = db.similarity_search(query)
    chain2 = LLMChain(llm = Config.llm, prompt = chat_prompt_2)
    ans = chain2.run({"output": docs[0], "question": query})
    return ans


""" f = True
while f == True:
    c = input("Do you have any questions? (Y/N): ")
    #print(c.lower)
    if c in ("Y", "y"):
        query = input("Enter question: ")
        docs = db.similarity_search(query)
        chain2 = LLMChain(llm = Config.llm, prompt = chat_prompt_2)
        print(chain2.run({"output": docs[0], "question": query}))

    else:
        print("Thanks!")
        f = False
        break"""