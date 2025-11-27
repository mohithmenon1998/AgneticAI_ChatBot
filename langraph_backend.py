from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from typing import TypedDict, Literal, Annotated
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(model= "llama3.2:3b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {'messages': [response]}

check_pointer = InMemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=check_pointer)