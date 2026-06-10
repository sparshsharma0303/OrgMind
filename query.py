from llama_index.core import VectorStoreIndex, StorageContext,StorageContext, load_index_from_storage,Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from src_logging.logger import logging
from exception import OrgMindException
import sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from pathlib import Path

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

        storage_path = Path("storage")
        doc_folders = [f for f in storage_path.iterdir() if f.is_dir()]
        if not doc_folders :
            raise Exception("No indexed documents found in storage/")

        indexes = []
        for doc_folder in doc_folders:
            doc_name = doc_folder.name        

            vector_store = FaissVectorStore.from_persist_dir(str(doc_folder/"vector"))

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(doc_folder / "vector")
            )
            vector_index = load_index_from_storage(storage_context=storage_context)

            logging.info("Vector index loaded for: %s", doc_name)
            
            # load tree index
            storage_context = StorageContext.from_defaults(
                persist_dir=str(doc_folder / "tree")
            )
            tree_index = load_index_from_storage(storage_context=storage_context)

            logging.info("Tree index loaded for: %s", doc_name)
            
            indexes.append((doc_name, vector_index, tree_index))
        logging.info('tree index loaded successfully ')

        return indexes
    
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
    

def retrieve_answer(query, indexes:list,chat_history:list):
    try:
        route = route_query(query)
        answers = []
        logging.info(f'successfully retrived route, route : {route}')

        for doc_name, vector_index, tree_index in indexes:


            Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
            # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

            if (route == "specific"):
                query_engine = tree_index.as_query_engine()
                response = query_engine.query(query)
                answers.append(f"[{doc_name}]: {response}")
                logging.info(f'successfully generated response using tree index')


            else:
                query_engine = vector_index.as_query_engine()
                response = query_engine.query(query)
                answers.append(f"[{doc_name}]: {response}")
                logging.info(f'successfully generated response using vector index')

        client = GroqClient(api_key = groq_api_key)
        messages = [
            {
                "role" : "system",
                "content": "You are given answers from multiple documents. Consolidate them into one clear, accurate answer. Remove duplicates. If answers conflict, mention both. At the end always add a line: 'Sources: [list the document names used]'"
            }
        ]
        if chat_history :
            messages.extend(chat_history)

        messages.append({
                        "role": "user",
                        "content": f"Query: {query}\n\nAnswers from documents:\n" + "\n\n".join(answers)
                        })
        fusion_response = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages  = messages,
            temperature= 0
        )

        fused = fusion_response.choices[0].message.content.strip()
        return format_answer(query, fused, route)

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


def ask(query,chat_history:list = None):
    try:
        if not chat_history :
            chat_history = []
        
        indexes = load_indexes()
        answer = retrieve_answer(query=query, indexes=indexes,chat_history=chat_history)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        return answer,chat_history
    except Exception as e:
        raise OrgMindException(str(e), sys)


if __name__ == "__main__":
    answer1, history = ask("What is the penalty for ethics violation?")
    print(answer1)
    print("---")
    
    answer2, history = ask("What happens if it is repeated?", chat_history=history)
    print(answer2)
