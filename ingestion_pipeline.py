import os
from  langchain_community.document_loaders import  TextLoader , DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()



def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} doesnt exist!")

    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents=loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"There are no .txt files here {docs_path}")

    return documents




def split_documents(documents,chunk_size=1000,chunk_overlap=150):
    print("Splitting documents into chuncks...")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )    

    chunks =text_splitter.split_documents(documents)

    return chunks



def create_vector_store(chunks,persist_directory="db/chroma_db"):
    print("Creating embedding and storing in chromaDB....")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    print("--create vector store--")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore



def main():
    documents=load_documents(docs_path="docs")
    chunks=split_documents(documents=documents)
    vectorstore=create_vector_store(chunks=chunks)


if __name__ == "__main__":
    main()