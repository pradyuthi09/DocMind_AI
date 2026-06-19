from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

def build_chain(vectordb):

    retriever = vectordb.as_retriever(
        search_kwargs={"k":4}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return chain