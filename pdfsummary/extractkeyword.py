from langchain.schema import BaseOutputParser
from config import Config
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import vector_embedding as ve
import templates as t
import user_input
from pathlib import Path

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")
    

system_message_prompt = SystemMessagePromptTemplate.from_template(t.template)
human_template = "{text}" + "{language}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm = Config.llm, prompt = chat_prompt)
#data = open()
path_pdf = user_input.get_path()
loader = UnstructuredPDFLoader(path_pdf)
pages = loader.load()

l = ""
l = input("Enter a Language if translation is required else press enter: ")
if l != "":
    print(chain.run({'text': pages[0], 'language': l}))
else:
    print(chain.run({'text': pages[0], 'language': 'english'}))
#print(pages[0].page_content)


system_message_prompt_2 = SystemMessagePromptTemplate.from_template(t.system_template)
human_message_prompt_2 = HumanMessagePromptTemplate.from_template(t.template2)
chat_prompt_2 = ChatPromptTemplate.from_messages([system_message_prompt_2, human_message_prompt_2])



#check if embedding dir is created else we create the collection
collection = ve.check_embedding_dir()

#check if embedding dir is empty
collection_is_empty = ve.check_empty(collection)
if collection_is_empty:
    documents = ve.text_splitter(pages)
    db = ve.apply_embedding_func(documents)
    db.persist()
else:
    db = collection.get()

f = True
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
        break


