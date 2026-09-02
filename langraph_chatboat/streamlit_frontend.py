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

  try:
    response = chatboat.invoke(
        {"message":[HumanMessage(content=user_input)]},
        config=CONFIG
    )

    ai_message = response["message"][-1].content

  except Exception as e:
    st.error(str(e))
  message_history.append({"role": "assistant", "content": ai_message})
  with st.chat_message("assistant"):
    st.write(ai_message)
