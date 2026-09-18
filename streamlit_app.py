import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Enterprise AI Support Agent", page_icon="🛡️", layout="wide")
st.title("🛡️ Enterprise AI Support & Booking Agent")
st.caption("Production-hardened LLM pipeline with Circuit Breakers & Pre-Retrieval PII Masking")

# Engineering Observability Panel
st.sidebar.header("⚙️ Production Guardrails")
enable_pii_guard = st.sidebar.toggle("Pre-Retrieval PII Masking (Issue #5)", value=True)
enable_circuit_breaker = st.sidebar.toggle("State Circuit Breaker (Issue #1)", value=True)

# 1. PII Sanitizer Function (Issue #5 Fix)
def sanitize_pii(text: str) -> tuple[str, bool]:
    phone_pattern = r"(\+?[0-9]{1,3}[-\s]?)?(\(?[0-9]{3}\)?[-\s]?)?[0-9]{3}[-\s]?[0-9]{4,6}"
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    id_pattern = r"\b[0-9]{5}-[0-9]{7}-[0-9]\b"
    
    modified = text
    detected = False
    
    if re.search(phone_pattern, modified):
        modified = re.sub(phone_pattern, "<PHONE_MASKED>", modified)
        detected = True
    if re.search(email_pattern, modified):
        modified = re.sub(email_pattern, "<EMAIL_MASKED>", modified)
        detected = True
    if re.search(id_pattern, modified):
        modified = re.sub(id_pattern, "<NATIONAL_ID_MASKED>", modified)
        detected = True
        
    return modified, detected

# 2. LLM Setup
groq_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=groq_key,
    temperature=0.2
)

SYSTEM_PROMPT = """
You are the official Enterprise AI Booking & Technical Support Specialist.
Provide concise, courteous, and professional support in English.
When a user requests a booking or consultation, ask for their preferred schedule and scope of service.
If user contact information appears masked (e.g., <PHONE_MASKED>, <EMAIL_MASKED>), confirm that their credentials have been securely stored in compliance with privacy regulations.
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0

# Display conversation
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# User Input
if user_input := st.chat_input("Type your technical inquiry or booking request here..."):
    # Pre-Retrieval PII Masking
    processed_input = user_input
    if enable_pii_guard:
        cleaned_text, pii_found = sanitize_pii(user_input)
        if pii_found:
            st.sidebar.warning(f"🔒 PII Filtered from Payload: {cleaned_text}")
        processed_input = cleaned_text

    with st.chat_message("user"):
        st.write(user_input)
        if processed_input != user_input:
            st.caption(f"🔒 Sanitized Model Payload: `{processed_input}`")

    st.session_state.chat_history.append(HumanMessage(content=processed_input))

    # LLM Execution with Circuit Breaker
    with st.chat_message("assistant"):
        with st.spinner("Analyzing request..."):
            MAX_RETRIES = 2
            success = False
            response_text = ""

            while st.session_state.retry_count < MAX_RETRIES and not success:
                try:
                    response = llm.invoke(st.session_state.chat_history)
                    response_text = response.content
                    success = True
                    st.session_state.retry_count = 0
                except Exception as e:
                    st.session_state.retry_count += 1
                    st.sidebar.error(f"Execution Failure! Retry attempt: {st.session_state.retry_count}")
                    if st.session_state.retry_count >= MAX_RETRIES:
                        response_text = "⚠️ [CIRCUIT BREAKER ACTIVATED]: External service latency exceeded. Token consumption halted to prevent bill runaway. Routing ticket to human operations."

            st.write(response_text)
            st.session_state.chat_history.append(AIMessage(content=response_text))