import streamlit as st 
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': '1'}}

session = st.session_state

if "message_history" not in session:
    
    session["message_history"] = []

for message in session["message_history"]:
    
    with st.chat_message(message["role"]):
        st.text(message["messages"])
    

user_input = st.chat_input("type here")

if user_input:
    
    with st.chat_message('user'):
        
        st.text(user_input)
    
    session["message_history"].append({"role": "user", "messages": user_input})
    
        
    with st.chat_message('assistant'):
        
        AI_Messages = st.write_stream(message_chunk.content for message_chunk, metadata in chatbot.stream({'messages':[HumanMessage(content= user_input)]}, stream_mode= "messages", config= CONFIG))
        
    session["message_history"].append({"role": "assistant", "messages": AI_Messages})
        