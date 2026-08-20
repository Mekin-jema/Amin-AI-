import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from chatbot_workflow import workflow
import uuid

# Set up page configurations
st.set_page_config(
    page_title="Amin AI - Chatbot UI",
    page_icon="🤖",
    layout="wide"
)

# App Title & Description
st.title("🤖 Amin AI - Chatbot Workspace")
st.markdown("Interact with your stateful LangGraph agent in real-time. Conversational memory is preserved via the InMemorySaver.")

# Initialize session state for thread_id and chat history
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Allow user to see/modify thread ID
    thread_id_input = st.text_input("Thread ID (Session identifier)", value=st.session_state.thread_id)
    if thread_id_input != st.session_state.thread_id:
        st.session_state.thread_id = thread_id_input
        # Clear local UI messages when switching threads
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Thread & Clear Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("- FastAPI & LangGraph")
    st.markdown("- LangChain & OpenRouter")

# Display conversation messages from session state
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Accept user chat input
if prompt := st.chat_input("Enter your message..."):
    # Create HumanMessage object
    user_message = HumanMessage(content=prompt)
    
    # Add to UI state and display immediately
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Prepare inputs for LangGraph workflow
    initialize_state = {
        "messages": [user_message]
    }
    
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }
    
    # Invoke workflow and display assistant response with streaming
    with st.chat_message("assistant"):
        def response_generator():
            # Use LangGraph's messages streaming mode to get token chunks
            for chunk, metadata in workflow.stream(
                initialize_state,
                config=config,
                stream_mode="messages"
            ):
                # Ensure the chunk comes from the chat_node and contains text content
                if metadata.get("langgraph_node") == "chat_node" and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        try:
            full_response = st.write_stream(response_generator())
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"Error invoking workflow: {str(e)}")
