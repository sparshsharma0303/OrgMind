from llama_index.core import VectorStoreIndex, StorageContext,StorageContext, load_index_from_storage,Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from src_logging.logger import logging
from exception import OrgMindException
import sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq as GroqClient
# from langchain_classic import chains

groq_api_key = os.getenv('GROQ_API_KEY')


def load_indexes():
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        vector_store = FaissVectorStore.from_persist_dir("storage/vector")
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="storage/vector")
        vector_index = load_index_from_storage(storage_context=storage_context)
        logging.info('vector index loaded successfully ')
        storage_context = StorageContext.from_defaults(persist_dir= "storage/tree")
        tree_index = load_index_from_storage(storage_context= storage_context)
        logging.info('tree index loaded successfully ')

        return vector_index, tree_index
    
    except Exception as e:
        raise OrgMindException(str(e),sys)
    

def route_query(query:str):
    try:
        client = GroqClient(api_key= os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """Classify the query as either 'specific' or 'broad'.
                    - specific: asks for a precise fact, rule, policy, or procedure
                    - broad: asks for a general overview, summary, or concept
                    Answer with only one word: specific or broad"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature= 0
            )
        route = response.choices[0].message.content.strip().lower()
        logging.info("Query routed as: %s", route)
        return route


    except Exception as e:
        raise OrgMindException(str(e),sys)
    

def retrieve_answer(query, vector_index, tree_index):
    try:
        route = route_query(query)
        logging.info(f'successfully retrived route, route : {route}')

        Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        if (route == "specific"):
            query_engine = tree_index.as_query_engine()
            response = query_engine.query(query)
            logging.info(f'successfully generated response using tree index')


        else:
            query_engine = vector_index.as_query_engine()
            response = query_engine.query(query)
            logging.info(f'successfully generated response using vector index')

        return format_answer(query, response, route)

    except Exception as e:
        raise OrgMindException(str(e),sys)
    
def format_answer(query, response, route):
    formatted = (
        f"Query: {query}\n"
        f"Route: [{route.upper()}]\n"
        f"Answer: {response}"
    )
    logging.info("Answer formatted for query: %s", query)
    return formatted


def ask(query):
    try:
        vector_index, tree_index = load_indexes()
        answer = retrieve_answer(query=query, vector_index=vector_index, tree_index=tree_index)
        return answer
    except Exception as e:
        raise OrgMindException(str(e), sys)


if __name__ == "__main__":
    print(ask("What is the penalty for ethics violation?"))
    print(ask("Tell me about AIESEC values"))
