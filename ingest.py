from pathlib import Path
import subprocess
import sys
import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings,TreeIndex
from llama_index.llms.groq import Groq
from src_logging.logger import logging
from exception import OrgMindException
from pathlib import Path



def validate_pdf(file_path):
    path = Path(file_path)
    if path.exists() : logging.info("File exists: %s", file_path)
    else: 
        logging.error("PDF not found: %s", file_path)
        return 
        
    if path.suffix == ".pdf":
            logging.info("Valid PDF file: %s", file_path)
            return True
    else :
            logging.error("Not a PDF file: %s", file_path)
            return False


def load_document(file_path:str):
    if not validate_pdf(file_path):
        logging.error("Document validation failed: %s", file_path)
        return
    documents = SimpleDirectoryReader(input_files =[file_path]).load_data()
    return documents
    

def build_vector_index(documents):
    '''
    takes the loaded documents and builds a FAISS vector index using LlamaIndex
    '''

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None
    faiss_index = faiss.IndexFlatL2(384) # FAISS index-384 is the embedding dimension 
    vector_store = FaissVectorStore(faiss_index = faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    logging.info("Vector index built successfully")
    return index


def build_tree_index(documents: list):
    '''
    summarizes document content into a hierarchy
    '''
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    index = TreeIndex.from_documents(documents)
    logging.info("Tree index built successfully")
    return index


def save_indexes(vector_index, tree_index,doc_name:str):
    try:
        vector_index.storage_context.persist(persist_dir=f"storage/{doc_name}/vector")
        tree_index.storage_context.persist(persist_dir=f"storage/{doc_name}/tree")
        logging.info("Indexes saved to disk")

    except Exception as e:
        raise OrgMindException(str(e),sys)

def index_document(file_paths: list):
    for file in file_paths:
            docs = load_document(file_path= file)
            vector_index = build_vector_index(docs)
            tree_index = build_tree_index(docs)
            doc_name = Path(file).stem.replace(" ", "_")
            save_indexes(vector_index=vector_index, tree_index=tree_index, doc_name=doc_name)
            
        


if __name__ == "__main__":
    index_document(["docs/B_Code of Ethics.pdf","docs/A_Brand Standardization (Brand and IM).pdf","docs/C_Election and Selection Procedure.pdf"])



