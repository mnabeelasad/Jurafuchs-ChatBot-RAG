import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import QA_PROMPT_TEMPLATE  # Your prompt from config.py

# Load all environment variables from .env
load_dotenv()

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content}" for doc in docs)

def get_rag_chain():
    """
    Creates and returns a configured RAG chain and its retriever.
    This is called once when the FastAPI app starts.
    """
    try:
        # --- 1. INITIALIZE MODELS ---
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"]
        )

        llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            temperature=0.1,
            max_tokens=500,
            top_p=0.95,
            validate_base_url=False
        )

        # --- 2. CONNECT TO VECTOR STORE ---
        vector_store = AzureSearch(
            azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
            azure_search_key=os.environ["AZURE_AI_SEARCH_KEY"],
            index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"],
            embedding_function=embeddings.embed_query,
        )

        # --- 3. CREATE RETRIEVER ---
        # Note: Set this to "semantic_hybrid" if you are on a paid tier
        retriever = vector_store.as_retriever(
            search_type="hybrid", 
            k=5
        )

        # --- 4. CREATE THE RAG CHAIN ---
        prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print("--- RAG Chain and Retriever initialized successfully. ---")
        return rag_chain, retriever

    except KeyError as e:
        print(f"CRITICAL ERROR: Missing environment variable {e}")
        return None, None
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize RAG chain: {e}")
        return None, None