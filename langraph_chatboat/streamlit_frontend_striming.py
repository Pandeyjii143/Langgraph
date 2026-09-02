import streamlit as st
from langraph_backend import chatboat
from langchain_core.messages import HumanMessage

st.title("Chat with Langraph Chatboat")
CONFIG={'configurable':{'thread_id': 'thread-1'}}
message_history=st.session_state["chat_history"] = st.session_state.get("chat_history", [])
#loding the conversation history
for message in message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:

  #first add the message to message history
  message_history.append({"role": "user", "content": user_input})
  with st.chat_message("user"):
    st.write(user_input)
    response = chatboat.invoke(
        {"message":[HumanMessage(content=user_input)]},
        config=CONFIG
    )
    


  with st.chat_message("assistant"):

     ai_message = st.write_stream(
        message_chunk.content
        for message_chunk, metadata in chatboat.stream(
            {"message":[HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        )
     )

  message_history.append({"role": "assistant", "content": ai_message})
  
