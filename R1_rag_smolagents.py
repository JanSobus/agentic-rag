"""Example of agentic RAG with SmolAgents."""

import logging
import os

import dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, GradioUI, HfApiModel, Model, OpenAIServerModel, ToolCallingAgent, tool

dotenv.load_dotenv()

RAG_MODEL_ID = os.getenv("RAG_MODEL", "deepseek-r1:7b-8k")
CHAT_MODEL_ID = os.getenv("CHAT_MODEL", "qwen2.5:14b-instruct-8k")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "ollama")
DB_DIR = os.getenv("DB_DIR", "chroma_db")

logger = logging.getLogger("Agentic RAG - SmolAgents")
logging.basicConfig(level=logging.INFO)


class WrongModelSourceError(Exception):
    """Exception raised when the model source is invalid."""

    def __init__(self, model_source: str) -> None:
        """Print custom error message."""
        super().__init__(f"Invalid model source: {model_source}")


def get_model(model_id: str, model_source: str) -> Model:
    """Get a model from the model source."""
    if model_source == "ollama":
        return OpenAIServerModel(model_id=model_id, api_base="http://localhost:11434/v1", api_key="ollama")
    if model_source == "huggingface":
        return HfApiModel(model_id=model_id, api_key=HF_TOKEN)
    raise WrongModelSourceError(model_source)


rag_model = get_model(RAG_MODEL_ID, MODEL_SOURCE)
rag_agent = CodeAgent(tools=[], model=rag_model, add_base_tools=False, max_steps=2)


# Initialize HuggingFace embeddings
embeddings_engine = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}
)
vector_db = Chroma(
    embedding_function=embeddings_engine,
    persist_directory=DB_DIR,
)


@tool
def rag_with_agent(query: str) -> str:
    """Use the RAG agent to answer a question based on vector database content.

    This tool takes in a user query and searches the vector database for relevant content.
    The result is then passed to the RAG agent to answer the question based on the context.

    Args:
        query: The user query to answer.

    Returns:
        The answer to the user query.

    """
    # Retrieve the most relevant documents from the vector database
    doc_chunks = vector_db.similarity_search(query, k=3)

    # Convert the document chunks to a context string
    context = "\n".join([doc.page_content for doc in doc_chunks])

    # Create appropriate prompt for the RAG agent
    rag_prompt = f"""Based on the following context, answer the user's question. Be specific and concise.
    If you don't have enough information, answer with a better query to perform RAG with. Answer in the same language
    as the question.

    Context:
    {context}

    Question:
    {query}

    Answer:"""

    # Call the RAG agent with the prompt
    return str(rag_agent.run(task=rag_prompt, reset=False))


chat_model = get_model(CHAT_MODEL_ID, MODEL_SOURCE)
chat_agent = ToolCallingAgent(
    model=chat_model, tools=[rag_with_agent], max_steps=3, name="Chat Agent", add_base_tools=False
)


def main() -> None:
    """Start the agentic RAG chatbot."""
    logger.info("Starting Agentic RAG with SmolAgents")

    ui = GradioUI(chat_agent)
    ui.launch()


if __name__ == "__main__":
    main()
