import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from llama_parse import LlamaParse
from langchain_core.documents import Document as LangchainDocument

# Load environment variables
load_dotenv()
start_time = time.time()

def document_chunking_and_uploading_to_vectorstore(filepath, actual_file_name):
    """
    Process a document, extract text from images and tables using LlamaParse,
    and upload chunks to a Pinecone vector store with metadata.

    Args:
        filepath (str): Path to the document file.
        actual_file_name (str): Name of the file to include in metadata.

    Returns:
        str: Summary of processing statistics or None if an error occurs.
    """
    try:
        # Define namespace for vector store
        name_space = "Test-1"  # Generate unique ID

        # Initialize embeddings
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

        # Set up Pinecone vector store
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = "italian-pdf-docs"
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(embedding=embeddings, index=index, namespace=name_space)

        # Initialize LlamaParse with advanced parsing instructions
        parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            result_type="markdown", 
            system_prompt="Extract text from images using multimodal models and include it in the output. Parse tables and texts in images. accurately into markdown format.",
            verbose=True,
            language="it", 
            ocr=True, 

        )

        # Load documents using LlamaParse
        llama_documents = parser.load_data(filepath)

        # Convert LlamaIndex documents to Langchain documents
        documents = [
            LangchainDocument(page_content=doc.text, metadata=doc.metadata)
            for doc in llama_documents
        ]

        # Add filename to metadata and verify page numbers
        for i, doc in enumerate(documents, start=1):
            doc.metadata["page"] = i  # Assign page number (e.g., 1, 2, 3, ...)
            doc.metadata["filename"] = actual_file_name  # Add filename to metadata

        # Configure text splitter to preserve metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            add_start_index=True
        )

        # Split documents into chunks
        all_splits = text_splitter.split_documents(documents)

        # Add chunks to vector store
        vector_store.add_documents(documents=all_splits)

        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)

        # Prepare summary
        info = (
            f"Document Processing Summary:\n"
            f"- Filename: {actual_file_name}\n"
            f"- Pages Processed: {len(documents)}\n"
            f"- Chunks Created: {len(all_splits)}\n"
            f"- Processing Time: {processing_time} seconds"
        )
        return info

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# # Example usage (uncomment to test)
# filepath = "path/to/your/document.pdf"
# actual_file_name = "document.pdf"
# result = document_chunking_and_uploading_to_vectorstore(filepath, actual_file_name)
# if result:
#     print(result)
