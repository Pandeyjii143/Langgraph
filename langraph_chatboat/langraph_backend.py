from langgraph.graph import StateGraph ,START,END
from langchain_openai import ChatOpenAI
from typing import TypedDict,Annotated,Literal
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import operator
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

from langgraph.graph.message import add_messages
class ChatState(TypedDict):
  message:Annotated[list[BaseMessage],add_messages]
llm=ChatOpenAI()

def chat_node(state: ChatState):
  #take user query from state 
  message=state['message']
  #send to llm 
  response=llm.invoke(message)
  #resoponse store in state
  return {'message':[response]}

checkpointer=MemorySaver()
graph=StateGraph(ChatState)
#node
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatboat = graph.compile(checkpointer=checkpointer)


chatboat.stream(
   {'message':[HumanMessage(content="Hello")]},
   config={"configurable":{"thread_id":'thread-1'}},
   stream_mode='messages'
)

print("Backend imported")