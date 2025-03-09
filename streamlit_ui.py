"""Alternative streamlit UI for the RAG agent."""
import streamlit as st

from R1_rag_smolagents import chat_agent


def init_chat_history()->None:
    """Initialize the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history()->None:
    """Display the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str)->None:
    """Handle the user input."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        response = chat_agent.run(prompt, reset=False)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def display_sidebar()->None:
    """Display the sidebar."""
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This Q&A bot uses RAG (Retrieval Augmented Generation) to answer questions about your documents.

        The process:
        1. Your query is used to search through document chunks
        2. Most relevant chunks are retrieved
        3. A reasoning model generates an answer based on the context
        """)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

def main()->None:
    """Launch the application."""
    # Set up Streamlit page
    st.set_page_config(page_title="Document Q&A Bot", layout="wide")
    st.title("Document Q&A Bot")

    # Initialize chat history
    init_chat_history()

    # Display chat interface
    display_chat_history()
    display_sidebar()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        handle_user_input(prompt)

if __name__ == "__main__":
    main()
