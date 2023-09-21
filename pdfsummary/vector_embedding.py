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
import chromadb
from pathlib import Path



#check if embedding dir exists else create
def check_embedding_dir():
    file_path = Path(__file__)
    client = chromadb.Client()
    #client.list_collections()
    collection = Path(file_path).parent.name
    c = client.get_or_create_collection(name = collection)
    return c    

# if embedding dir is empty
def text_splitter(doc_to_split):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(doc_to_split)
    return documents

def apply_embedding_func(documents):
    emb_func = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, emb_func, persist_directory = "./chromadb")
    return db

#check if embedding dir is empty
def check_empty(collection):
    if collection.count() == 0:
        return True
    else:
        return False
    

