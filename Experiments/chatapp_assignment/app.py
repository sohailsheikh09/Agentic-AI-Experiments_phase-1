from typing import TypedDict, Annotated
import os
import sqlite3

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# =========================================================
# Load environment variables
# =========================================================
load_dotenv(override=True)


# =========================================================
# LLM Setup
# =========================================================
def get_groq_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=2000,
    )


# =========================================================
# Database / Memory Setup
# =========================================================
sqlite_conn = sqlite3.connect(
    "bot_checkpoint.sqlite",
    check_same_thread=False
)

memory = SqliteSaver(sqlite_conn)
llm = get_groq_llm()


# =========================================================
# State Definition
# =========================================================
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


# =========================================================
# Chatbot Node
# =========================================================
def chatbot(state: BasicChatState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }


# =========================================================
# Build Graph
# =========================================================
graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)
graph.set_entry_point("chatbot")

app_graph = graph.compile(checkpointer=memory)




# =========================================================
# Helper Functions
# =========================================================
def get_all_threads():
    """
    Retrieve all thread IDs stored in the SQLite checkpoint database.
    """
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute(
            """
            SELECT DISTINCT thread_id
            FROM checkpoints
            ORDER BY thread_id DESC
            """
        )
        rows = cursor.fetchall()
        return [str(row[0]) for row in rows]
    except Exception:
        return []



def get_messages_for_thread(thread_id):
    """
    Load latest messages for a thread.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = app_graph.get_state(config)
        if state and state.values and "messages" in state.values:
            return state.values["messages"]
    except Exception:
        pass

    return []




def create_new_thread_id():
    import uuid
    return str(uuid.uuid4())[:8]



def delete_thread(thread_id):
    """
    Optional: delete all checkpoints for a thread.
    """
    cursor = sqlite_conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,),
        )
        sqlite_conn.commit()
    except Exception as e:
        st.error(f"Error deleting thread: {e}")


# =========================================================
# Streamlit UI Setup
# =========================================================
st.set_page_config(
    page_title="LangGraph Chat",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangGraph Chat Assistant")

# =========================================================
# Session State Initialization
# =========================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_new_thread_id()

if "loaded_messages" not in st.session_state:
    st.session_state.loaded_messages = []


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.thread_id = create_new_thread_id()
        st.session_state.loaded_messages = []
        st.rerun()

    threads = get_all_threads()

    if threads:
        selected_thread = st.selectbox(
            "Select previous conversation",
            options=threads,
            index=0 if st.session_state.thread_id not in threads else threads.index(st.session_state.thread_id),
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Load"):
                st.session_state.thread_id = selected_thread
                st.session_state.loaded_messages = get_messages_for_thread(selected_thread)
                st.rerun()

        with col2:
            if st.button("Delete"):
                delete_thread(selected_thread)
                if st.session_state.thread_id == selected_thread:
                    st.session_state.thread_id = create_new_thread_id()
                    st.session_state.loaded_messages = []
                st.rerun()

    st.markdown("---")
    st.write(f"**Current Thread:** `{st.session_state.thread_id}`")



# =========================================================
# Load Messages from Current Thread
# =========================================================
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

messages = get_messages_for_thread(st.session_state.thread_id)


# =========================================================
# Display Chat Messages
# =========================================================
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# =========================================================
# User Input
# =========================================================
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app_graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)]
                },
                config=config,
            )

            ai_response = result["messages"][-1].content
            st.markdown(ai_response)

    st.rerun()