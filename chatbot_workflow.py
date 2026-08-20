from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 

# Initialize the state
class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the LLM
llm = ChatOpenAI(
    # model="openai/gpt-oss-20b",
    # GPT-4 family
    # model="openai/gpt-4o-mini"       # 128K context | 16K output
    # model="openai/gpt-4o"            # 128K context | 16K output
    # model="openai/gpt-4.1"           # 1M  context   | 32K output
    model="openai/gpt-4.1-mini"   ,   # 1M context   | 32K output

    # GPT-5 family
    # model="openai/gpt-5"             # 400K context | 128K output
    # model="openai/gpt-5-chat"        # 400K context | 128K output

    # Open-weight models
    # model="openai/gpt-oss-20b"       # 131K context | 131K output
    # model="openai/gpt-oss-20b:free"  # 131K context | 131K output
    # model="openai/gpt-oss-120b"      # 131K context | 131K output

    
    temperature=0,
    timeout=30,
    max_retries=0,
    max_tokens=2506,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

# Initialize the checkpoint
checkpointer = InMemorySaver()

# Build the graph
graph = StateGraph(chatState)

# Implement the node function
def chat_node(state: chatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
 
# Add node and edges
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile the workflow with checkpoint
workflow = graph.compile(checkpointer=checkpointer)

def main():
    # Invoke the graph
    thread_id = "thread-1" 
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        content=input("Enter your questions:")
        initialize_state = {
            "messages": [
                HumanMessage(content=content)
            ]
        }
        
        print("Invoking workflow...")
        result = workflow.invoke(initialize_state,config=config)
        # print(result)
        
        print("\nResult Messages:")
        for msg in result["messages"]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            print(f"{role}: {msg.content}")

if __name__ == "__main__":
    main()
