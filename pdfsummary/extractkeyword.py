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
from pathlib import Path
from prompt_toolkit import prompt


class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")
        


def get_db_pages(path_pdf):
    #path_pdf = user_input.get_path_pdf()
    db, pages = asyncio.run(ve.init_vector_store(path_pdf))
    return db, pages


#to find summary
async def summary_llm(pages):
    text = ve.convert_to_text(pages)
    system_message_prompt = SystemMessagePromptTemplate.from_template(t.summary_template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm = Config.llm, prompt = chat_prompt)
    summary = await chain.arun({'text': text})
    return summary

async def translation(summary, language):
    system_message_prompt_3 = SystemMessagePromptTemplate.from_template(t.system_translate_template)
    human_message_prompt_3 = HumanMessagePromptTemplate.from_template(t.translate_template)
    chat_prompt_3 = ChatPromptTemplate.from_messages([system_message_prompt_3, human_message_prompt_3])
    chain3 = LLMChain(llm = Config.llm, prompt = chat_prompt_3)
    translate =await chain3.arun({'summary': summary, 'language': language})
    return translate
   
#print(pages[0].page_content)

#template for question answer
async def question_llm(query, db):
    system_message_prompt_2 = SystemMessagePromptTemplate.from_template(t.system_template)
    human_message_prompt_2 = HumanMessagePromptTemplate.from_template(t.template2)
    chat_prompt_2 = ChatPromptTemplate.from_messages([system_message_prompt_2, human_message_prompt_2])
    docs = db.similarity_search(query)
    ve.convert_to_text(docs)
    chain2 = LLMChain(llm = Config.llm, prompt = chat_prompt_2)
    ans = await chain2.arun({"output": docs, "question": query})
    return ans


if __name__ == "__main__":
    import asyncio
    path_pdf = Path("C:/Users/Sayalee/Documents/langchain_research_paper.pdf")
    db, pages = asyncio.run(ve.init_vector_store(path_pdf))
    print(asyncio.run(summary_llm(pages)))
    #print(pages)