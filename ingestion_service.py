import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

def process_and_ingest_file(file_path: str, index_name: str):
    """
    Loads a PDF from the given path, splits it, and ingests it
    into the specified Azure AI Search index.
    """
    
    # --- 1. CONFIGURE EMBEDDING MODEL ---
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["OPENAI_API_VERSION"]
    )

    # --- 2. CONFIGURE AZURE AI SEARCH (Vector Store) ---
    vector_store = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        azure_search_key=os.environ["AZURE_AI_SEARCH_KEY"],
        index_name=index_name,  # Use the index_name passed to the function
        embedding_function=embeddings.embed_query,
    )

    # --- 3. LOAD DOCUMENT ---
    print(f"Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # --- 4. TEXT SPLITTING (Chunking) ---
    print(f"Loaded {len(documents)} pages. Now splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # --- 5. EMBED AND INGEST ---
    print(f"Adding documents to Azure AI Search index: {index_name}...")
    vector_store.add_documents(documents=chunks)

    print(f"Successfully added {len(chunks)} chunks to the index '{index_name}'.")
    return len(chunks)