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



#check if embedding dir exists
def check_embedding_dir(collection):
    client = chromadb.Client()
    try:
        c = client.get_collection(collection)
        if len(c) > 0:
            return True
        else:
            return False
    except:
        return False

        


def text_splitter(doc_to_split):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(doc_to_split)
    emb_func = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, emb_func, persist_directory = "./chromadb")
    db.persist()