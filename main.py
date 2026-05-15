
import os
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Setup
groq_key = os.getenv("GROQ_API_KEY")
hf_token  = os.getenv("HF_TOKEN")

groq_client = Groq(api_key=groq_key)
duck_search = DuckDuckGoSearchRun()

# Load Vectorstore
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_token,
)

vectorstore = FAISS.load_local(
    "data/vectorstore",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Contact Info
CONTACT_INFO = """
Countryside Resort Gilgit - Contact Information:
Phone: 03109152096 / 03555033760
Email: Countrysideresortglt@gmail.com
Location: Jageer Baseen, Gilgit (6km from airport)
Website: https://www.countrysideresortgilgit.com
Social: @Countrysideresortgilgit (Facebook & Instagram)
Bank: Habib Metropolitan Bank
IBAN: PK61 MPBL 0286 0271 4013 9401
"""

SYSTEM_PROMPT = """You are Ali, a warm and helpful guest assistant for Countryside Resort Gilgit, located in beautiful Gilgit-Baltistan, Pakistan.
Answer guests warmly and in detail based only on the context provided.
Always respond in English. Be enthusiastic about Gilgit-Baltistan."""

# Memory Store
conversation_store = {}

# Context Router
def get_context(question):
    q = question.lower()

    if any(w in q for w in ["phone", "email", "contact", "book",
                              "reservation", "payment", "iban", "bank"]):
        return CONTACT_INFO, "contacts"

    elif any(w in q for w in ["weather now", "weather today", "road condition",
                               "road open", "flight status", "highway"]):
        try:
            return duck_search.run(f"{question} Gilgit-Baltistan Pakistan"), "web"
        except:
            return "Web search unavailable.", "web"

    else:
        results = retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in results]), "resort_docs"

# Core Agent
def run_agent(question, thread_id="guest-1"):
    context, source = get_context(question)

    if thread_id not in conversation_store:
        conversation_store[thread_id] = []

    history = conversation_store[thread_id]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]
    messages.append({
        "role": "user",
        "content": f"""Guest question: {question}

Relevant information:
{context}

Give a warm, complete and helpful answer based on the above."""
    })

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    conversation_store[thread_id].append({"role": "user", "content": question})
    conversation_store[thread_id].append({"role": "assistant", "content": answer})

    return answer, source

# FastAPI App
app = FastAPI(title="Countryside Resort Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    thread_id: str = "guest-1"

class ChatResponse(BaseModel):
    answer: str
    source: str

@app.get("/")
def root():
    return {"status": "Countryside Resort Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer, source = run_agent(request.question, request.thread_id)
    return ChatResponse(answer=answer, source=source)

@app.delete("/chat/{thread_id}")
def clear_history(thread_id: str):
    if thread_id in conversation_store:
        del conversation_store[thread_id]
    return {"status": f"History cleared for {thread_id}"}
