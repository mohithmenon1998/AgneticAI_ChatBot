from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from typing import TypedDict, Literal, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()

ddg_tool= DuckDuckGoSearchRun()

@tool(name_or_callable= "weather_tool", description= "Get realtime weather information of any city")
def weather_tool(city: str):
    "Tool to get realtime weather information of any city, argument required is the city in str format"
    url = f"http://api.weatherapi.com/v1/current.json?key=135e5756efaf4dcda80183250252411&q={city}"
    r = requests.get(url= url)
    return r.json()

tools = [ddg_tool, weather_tool]

model = ChatOllama(model= "llama3.2:3b").bind_tools(tools= tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState):
    f"""Answer the following questions as best you can. You have access to the following tools:\n\n{tools}"""
    messages = state["messages"]
    response = model.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools=tools)
conn = sqlite3.connect(database="chatbot.db", check_same_thread= False)
check_pointer = SqliteSaver(conn= conn)


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=check_pointer)

def retrieve_threads():
    all_threads = set()
    for checkpoint in check_pointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
        
    return(list(all_threads))