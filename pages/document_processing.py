def document_chunking_and_uploading_to_vectorstore(filepath, actual_file_name):
    try:
        name_space = "Uploaded_document"
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = "italian-pdf-docs"
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(embedding=embeddings, index=index, namespace=name_space)

        filename = os.path.basename(filepath)
        loader = PyPDFLoader(filepath)

        async def load_pages(loader):
            pages = []
            async for page in loader.alazy_load():
                page.metadata['page'] = page.metadata['page'] + 1
                page.metadata["filename"] = actual_file_name
                print(f"Loaded page {page.metadata['page']} with metadata: {page.metadata}")
                pages.append(page)
            return pages

        docs = asyncio.run(load_pages(loader))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        for split in all_splits:
            if 'page' not in split.metadata:
                print(f"Warning: page number missing in split metadata: {split.metadata}")
            if 'filename' not in split.metadata:
                split.metadata["filename"] = filename

        vector_store.add_documents(documents=all_splits)
        processing_time = round(time.time() - time.time(), 3)  # Fix: Move start_time inside function
        info = (
            f"Document Processing Summary:\n"
            f"- Filename: {actual_file_name}\n"
            f"- Pages Processed: {len(docs)}\n"
            f"- Chunks Created: {len(all_splits)}\n"
            f"- Processing Time: {processing_time} seconds"
        )
        return True, info
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return False, error_message
