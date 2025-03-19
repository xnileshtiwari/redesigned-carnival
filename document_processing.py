import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings

load_dotenv()
start_time = time.time()

def document_chunking_and_uploading_to_vectorstore(filepath, actual_file_name):
    try:
        name_space = "Test-1" 

        embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0",
        )

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # get api key
        index_name = "italian-pdf-docs"  # get index name
        index = pc.Index(index_name)  # get index

        vector_store = PineconeVectorStore(embedding=embeddings, index=index, namespace=name_space)  # create vector store

        # Extract just the filename from the full filepath
        filename = os.path.basename(filepath)

        # Define document loader
        loader = PyPDFLoader(filepath)
        async def load_pages(loader):
            pages = []
            async for page in loader.alazy_load():
                # Adjust the page number to start from 1 instead of 0
                page.metadata['page'] = page.metadata['page'] + 1
                # Add extra metadata: source filename only, not the entire path
                page.metadata["filename"] = actual_file_name
                print(f"Loaded page {page.metadata['page']} with metadata: {page.metadata}")
                pages.append(page)
            return pages

        docs = asyncio.run(load_pages(loader))
        
        # Configure text splitter to preserve metadata during splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            add_start_index=True,  # This will help track chunk positions
        )
        
        all_splits = text_splitter.split_documents(docs)
        # Verify metadata in splits and ensure 'source' is present
        for split in all_splits:
            if 'page' not in split.metadata:
                print(f"Warning: page number missingn split metadata: {split.metadata}")
            if 'filename' not in split.metadata:
                split.metadata["filename"] = filename
        
        # Add documents to vector store with metadata
        vector_store.add_documents(documents=all_splits)
        
        # Calculate processing time rounded to 3 decimal places
        processing_time = round(time.time() - start_time, 3)
        
        # Return statistics as a formatted string
        info = (
            f"Document Processing Summary:\n"
            f"- Filename: {actual_file_name}\n"
            f"- Pages Processed: {len(docs)}\n"
            f"- Chunks Created: {len(all_splits)}\n"
            f"- Processing Time: {processing_time} seconds"
        )
        return info
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


# filepath = "Zoning Fedex International Contract.pdf"

# new_file = document_chunking_and_uploading_to_vectorstore(filepath=filepath)
# print(new_file)
