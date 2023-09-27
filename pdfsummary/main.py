import chainlit as cl
import extractkeyword as extractk
import vector_embedding as ve
from pathlib import Path

@cl.on_chat_start
async def start():

    path_pdf = None


    while path_pdf == None:
        path_pdf = await cl.AskUserMessage(
            content=f"enter path of PDF", timeout=15
        ).send()
       
    #after we get the Path
    db, pages = await ve.init_vector_store(Path(path_pdf['content']))
    summary = await extractk.summary_llm(pages)
    
    await cl.Message(
        content=f"The summary of the PDF file is: {summary}"
    ).send()

    ans = None
    lang = None
    while ans == None:

        ans = await cl.AskUserMessage(
            content=f"Do you want to translate the summary? (Yes/No)", timeout=15
        ).send()
    
    if ans['content'] in ("yes", "y", "Yes", "Y"):
        while lang == None:
            lang = await cl.AskUserMessage(
                content=f"Which Language do you want to use to translate the summary?", timeout=15
            ).send()
        translation = await extractk.translation(summary, lang['content'])

        await cl.Message(
            content=f"The translation of the summary file is: {translation}"
        ).send()
    

