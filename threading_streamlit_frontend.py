import streamlit as st 
# from tools_backend_langraph import chatbot, model, retrieve_threads
from v1_streamlit_backend import chatbot, retrieve_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


 # Initialize session state
session = st.session_state

# ---------utility functions------------

def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    session["thread_id"] = generate_thread_id()
    add_thread_id(session["thread_id"])
    session["message_history"] = []

def add_thread_id(thread_id):
    if thread_id not in session["chat_threads"]:
        session["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    
    return chatbot.get_state(config= {'configurable': {'thread_id': thread_id}}).values["messages"]
        
# Check if message history exists in session state
if "message_history" not in session:
    
    session["message_history"] = []
    
if "thread_id" not in session:
    
    session["thread_id"] = generate_thread_id()

if "chat_threads" not in session:
    
    session["chat_threads"] = retrieve_threads()
    
add_thread_id(session["thread_id"])
    
# ---------sidebar ui-------------
st.sidebar.title("MohuGPT")    

if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My convos")

for thread_id in session["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        session["thread_id"] = thread_id
        messages = load_conversation(thread_id=thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user" 
            elif isinstance(message, AIMessage):
                role = "assistant"
            temp_messages.append({"role": role, "messages": message.content})    
        session["message_history"] = temp_messages            
# ---------------main ui---------------------
# Display previous messages
for message in session["message_history"]:
    
    with st.chat_message(message["role"]):
        st.text(message["messages"])
    
# Get user input
user_input = st.chat_input("type here")

# ----------main logic----------
# Process user input
if user_input:
    # Display user message
    with st.chat_message('user'):
        
        st.text(user_input)
    # Append user message to history
    session["message_history"].append({"role": "user", "messages": user_input})
    
    # Get and display assistant response    
    CONFIG = {'configurable': {'thread_id': session["thread_id"]}, 'metadata': {'thread_id': session["thread_id"]}, 'run_name': 'chat turn' }
    with st.chat_message('assistant'):
        status_holder = {"box": None}
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())
        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )


    # Append assistant message to history    
    session["message_history"].append({"role": "assistant", "messages": ai_message})

if __name__ == "__main__":
    print("Streamlit LangGraph Frontend for MohuGPT")