import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

CSV_PATH = "../data/realistic_restaurant_reviews.csv"
COLLECTION_NAME = "restaurant_reviews"

vector_store: Chroma | None = None

def createVectorDb(modelName):
    db_path = f"./chrome_{modelName}_db"
    embeddings = OllamaEmbeddings(model=modelName)
    global vector_store 
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=db_path,
        embedding_function=embeddings
    )
    if not os.path.exists(db_path):
        ids = []
        documents = []
        df = pd.read_csv(CSV_PATH)
        for i, row in df.interrows():
            ids.append(str(i))
            documents.append(
                Document(
                    page_content=row["Title"] + " " + row["Review"],
                    metadata={"rating": row["Rating"], "date": row["Date"]},
                    id=str(i)
                )
            )
        vector_store.add_documents(documents=documents, ids=ids)
    print(f"Vector database is ready")

def getStatus():
    global vector_store 
    print(vector_store)

def getRetriever():
    global vector_store 
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever
