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




path_pdf = user_input.get_path_pdf()
loader = UnstructuredPDFLoader(path_pdf)
pages = loader.load()
#check if embedding dir is created else we create the collection
db = ve.init_vector_store()
if db == False:
    ve.text_splitter(pages)
    

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")
    



if __name__ == "__main__":
    
    #to find summary
    system_message_prompt = SystemMessagePromptTemplate.from_template(t.template)
    human_template = "{text}" + "{language}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm = Config.llm, prompt = chat_prompt)
    l = ""
    l = prompt("Enter a Language if translation is required else press enter: ")
    if l != "":
        print(chain.run({'text': pages, 'language': l}))
    else:
        print(chain.run({'text': pages, 'language': 'english'}))
    #print(pages[0].page_content)

    #template for question answer
    system_message_prompt_2 = SystemMessagePromptTemplate.from_template(t.system_template)
    human_message_prompt_2 = HumanMessagePromptTemplate.from_template(t.template2)
    chat_prompt_2 = ChatPromptTemplate.from_messages([system_message_prompt_2, human_message_prompt_2])

    f = True
    while f == True:
        c = prompt("Do you have any questions? (Y/N): ")
        #print(c.lower)
        if c in ("Y", "y"):
            query = prompt("Enter question: ")
            docs = db.similarity_search(query)
            chain2 = LLMChain(llm = Config.llm, prompt = chat_prompt_2)
            print(chain2.run({"output": docs[0], "question": query}))

        else:
            print("Thanks!")
            f = False
            break