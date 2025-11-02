from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from rag_service import get_rag_chain
from ingestion_service import process_and_ingest_file
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# --- Create 'data' folder if it doesn't exist ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize the FastAPI app
app = FastAPI(
    title="Jurafuchs RAG API",
    description="An API for querying documents using Azure AI Search and OpenAI."
)

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jurafuchs-chatbot-rag-1.onrender.com", 
        "http://127.0.0.1",
        "http://localhost"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for API ---
class ChatRequest(BaseModel):
    query: str
    index_name: str  # Now we specify which index to chat with

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class UploadResponse(BaseModel):
    message: str
    index_name: str
    chunks_added: int

# --- Global variable to hold reusable components ---
# We store the *functions* to create chains, not the chains themselves,
# since each chain is now specific to an index.
rag_chain_factory = None
retriever_factory = None

@app.on_event("startup")
def startup_event():
    """
    On app startup, create the RAG chain and retriever *factories*
    (functions that can create them on demand).
    """
    global rag_chain_factory, retriever_factory
    
    # This is a "factory" that creates a new chain when you call it
    # We pass the index_name in as a parameter now.
    def create_chain_and_retriever(index_name: str):
        # We can re-use the function from rag_service, but we need to
        # modify rag_service.py to accept index_name as an argument.
        # For simplicity, we'll redefine the logic here.
        
        from rag_service import get_rag_chain, format_docs
        from langchain_community.vectorstores import AzureSearch
        from langchain_openai import AzureOpenAIEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from config import QA_PROMPT_TEMPLATE
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        # 1. Get Models (from rag_service, or redefine here)
        # We need the embedding function to create the vector store client
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
        
        # 2. Connect to the *specific* vector store index
        vector_store = AzureSearch(
            azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
            azure_search_key=os.environ["AZURE_AI_SEARCH_KEY"],
            index_name=index_name,  # Use the specific index
            embedding_function=embeddings.embed_query,
        )
        
        # 3. Create Retriever
        retriever = vector_store.as_retriever(search_type="hybrid", k=5)
        
        # 4. Create RAG Chain
        prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain, retriever

    # Store the factory function itself in the global variable
    global chain_factory
    chain_factory = create_chain_and_retriever
    print("--- RAG Chain Factory initialized successfully. ---")


# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Receives a PDF file, saves it, ingests it into Azure AI Search,
    and returns a new unique index name for chatting.
    """
    try:
        # Create a unique index name from the file name
        # (Remove spaces, special chars, and add a simple hash)
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        index_name = f"doc-{safe_filename.lower().replace('.', '-')}"
        
        # Save the file temporarily
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and ingest the file into the new index
        chunks_added = process_and_ingest_file(file_path, index_name)
        
        # Optional: Delete the temp file after ingestion
        os.remove(file_path)

        return UploadResponse(
            message=f"File '{file.filename}' uploaded successfully.",
            index_name=index_name,
            chunks_added=chunks_added
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user's query AND an index_name,
    gets an answer from the correct RAG chain,
    and returns the answer and the sources.
    """
    if not chain_factory:
        raise HTTPException(status_code=503, detail="RAG service is not available.")
    
    if not request.index_name:
        raise HTTPException(status_code=400, detail="No index_name provided.")

    try:
        # Create a RAG chain *on-the-fly* for the requested index
        rag_chain, retriever = chain_factory(request.index_name)
        
        # 1. Get the answer
        answer = rag_chain.invoke(request.query)
        
        # 2. Get the source documents
        source_docs = retriever.invoke(request.query)
        
        # 3. Format the sources
        unique_sources = set(f"Page {doc.metadata.get('page', 'N/A')}" for doc in source_docs)
        
        return ChatResponse(answer=answer, sources=sorted(list(unique_sources)))

    except Exception as e:
        # This often happens if the index is still being created
        if "index_name" in str(e):
             raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found or not ready. Please wait a moment.")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Jurafuchs RAG API. Go to /docs for details."}