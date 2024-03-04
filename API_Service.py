from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.embeddings import HuggingFaceEmbeddings

# Import your llamacpp model
from llama_cpp import Llama
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain


# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define request body model
class Query(BaseModel):
    query: str

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Instantiate your llamacpp model
llm_model_path = "C:\\Users\\IAmTheWizard\\Documents\\GitHub\\stable-diffusion-webui\\extensions\\llama2Training-folder\\llama-2-13b-chat.Q5_K_M.gguf"
llm = LlamaCpp(
    model_path=llm_model_path,
    n_gpu_layers=20,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    verbose=True,
)

# Initialize prompt
prompt = PromptTemplate(template = template, input_variables = ['context', 'question'])

# Retrieve FAISS and and initiate retriever
loaded_embedding_model = SentenceTransformer("C:\\Users\\IAmTheWizard\\Documents\\GitHub\\stable-diffusion-webui\\extensions\\llama2Training-folder\\sentence_transformers")
faiss_path = "C:\\Users\\IAmTheWizard\\Documents\\GitHub\\stable-diffusion-webui\\extensions\\llama2Training-folder\\faiss"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device': 'cpu'})
db = FAISS.load_local(faiss_path, embeddings)

retriever = db.as_retriever(search_kwargs = {'k': 2})
#prompt = PromptTemplate(template = template, input_variables = ['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm = llm, chain_type = 'stuff', retriever = retriever, return_source_documents = True, chain_type_kwargs = {'prompt': prompt})


# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
#retriever = VectorStoreRetriever(vectorstore=db)
#retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)


# Define API endpoint
@app.post("/query/")
def query_endpoint(query: Query):
    
    time_start = time.time()
    
    # Generate response using the model
    response = chain({'query': query.query})

    time_elapsed = time.time() - time_start

    return {"response": response, "time elapsed": time_elapsed}

@app.post("/rag/query/")
def query_endpoint(query: Query):
    
    time_start = time.time()
    
    # Generate response using the model
    response = qa_llm({'query': query.query})

    time_elapsed = time.time() - time_start

    return {"response": response, "time elapsed": time_elapsed}