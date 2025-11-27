from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from typing import TypedDict, Literal, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(model= "llama3.2:3b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database="chatbot.db", check_same_thread= False)
check_pointer = SqliteSaver(conn= conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=check_pointer)

def retrieve_threads():
    all_threads = set()
    for checkpoint in check_pointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
        
    return(list(all_threads))