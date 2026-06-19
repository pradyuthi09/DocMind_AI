from langchain_community.vectorstores import Chroma

def create_vector_db(chunks, embeddings, persist_directory="chroma_db"):

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectordb