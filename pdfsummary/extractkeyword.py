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


class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")
    
template = """You display keywords and a summary.
A user will pass in a pdf file, you should generate keywords and a summary from it with labels 'Keywords: ' and 'Summary: ' on two different lines.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm = Config.llm, prompt = chat_prompt, output_parser = CommaSeparatedListOutputParser())
#data = open()

loader = UnstructuredPDFLoader(r"C:/Users/Sayalee/Documents/samplepdf1.pdf")
pages = loader.load()


print(chain.run(pages[0]))
