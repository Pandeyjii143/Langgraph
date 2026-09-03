import streamlit as st
from langraph_backend import chatboat
from langchain_core.messages import HumanMessage
import uuid

#******************************utility functions********************************
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["chat_history"] = []
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

def load_conversation_history(thread_id):
    return chatboat.get_state(config={'configurable': {'thread_id': thread_id}}).values['message']




#*****************************Title ************************************
st.title("Chat with Langraph Chatboat")





#****************************session Setup********************************

message_history=st.session_state["chat_history"] = st.session_state.get("chat_history", [])

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = []

add_thread(st.session_state["thread_id"])

#*************************************Sidebar UI ******************************************
st.sidebar.title("Langraph Chatboat")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.markdown("## Conversation History")

for thread_id in st.session_state['chat_thread']:
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id
        messages=load_conversation_history(thread_id)
        

        temp_message=[]
        for message in messages:
            if isinstance(message, HumanMessage):
                temp_message.append({"role": "user", "content": message.content})
            else:
                temp_message.append({"role": "assistant", "content": message.content})
        st.session_state["chat_history"] = temp_message
        st.rerun()

st.write("History Length:", len(st.session_state["chat_history"]))

   

#*********************************Main UI ******************************************


CONFIG={'configurable':{'thread_id': st.session_state["thread_id"]}}



#loding the conversation history
for message in message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:

    message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            chunk.content
            for chunk, metadata in chatboat.stream(
                {
                    "message": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            )
            if hasattr(chunk, "content")
        )

    message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    st.session_state["chat_history"] = message_history