import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Groq Model initialization
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

SYSTEM_INSTRUCTION = """
Aap Asani Tech ke official AI Booking & Support Assistant hain.
Aap customer ke sawalat ke mukhtasir, muaddabana aur wazeh jawabat Roman Urdu ya English mein dete hain.
Agar koi booking ya appointment ka poochay toh tareekh aur service ki tafseelat maangein.
"""

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [SystemMessage(content=SYSTEM_INSTRUCTION)])
    await cl.Message(content="Khushamdeed! Main Asani Tech ka AI Assistant hoon. Main aapki kya madad kar sakta hoon?").send()

@cl.on_message
async def on_message(message: cl.Message):
    messages = cl.user_session.get("messages")
    messages.append(HumanMessage(content=message.content))
    
    response = await llm.ainvoke(messages)
    
    messages.append(response)
    cl.user_session.set("messages", messages)
    
    await cl.Message(content=response.content).send()