import chainlit as cl
import extractkeyword as extractk
import vector_embedding as ve
from pathlib import Path



@cl.on_chat_start
async def start() -> cl.Message:

    path_pdf = None


    while path_pdf == None:
        path_pdf = await cl.AskUserMessage(
            content=f"enter path of PDF", timeout = 15
        ).send()
       
    #after we get the Path
    db, pages = await ve.init_vector_store(Path(path_pdf['content']))
    summary = await extractk.summary_llm(pages)
    
    await cl.Message(
        content=f"{summary}"
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
    else:
        await cl.Message(
                content=f"okay"
            ).send()
    
    choiceq= None
    
    while choiceq == None:
        choiceq = await cl.AskUserMessage(
                    content=f"Do you have any questions about this? (Yes/No)", timeout=15
                ).send()
        
    if choiceq['content'] in ("yes", "y", "Yes", "Y"):
        f = True
        while f == True:
            ques = None
            while ques == None:
                ques = await cl.AskUserMessage(
                        content=f"What is the Question?", timeout=15
                    ).send()
                
            answer = await extractk.question_llm(ques['content'], db)
            await cl.Message(
                content=f"{answer}"
            ).send()
            c = None
            while c == None:
                c = await cl.AskUserMessage(
                        content=f"Do you have more questions (Yes/No)", timeout=15
                    ).send()
            if c not in ("yes", "y", "Yes", "Y"):
                f = False

    await cl.Message(
                content=f"Thank You!"
            ).send()