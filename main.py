
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

SYSTEM_PROMPT = """You are Ali, a helpful assistant for Countryside Resort Gilgit, Gilgit-Baltistan, Pakistan.

RESPONSE RULES — follow strictly:

1. ALWAYS be brief and structured. Never write long paragraphs.

2. FORMAT by question type:

   ROOMS question → use this format:
   Brief intro (1 line)
   • Room Name — key feature
     Price: Low 16,000 | Base 20,000 | Peak 24,000 | Eid 28,000 PKR
   (list each room this way)
   End with 1 line about included amenities.

   ATTRACTIONS question → use this format:
   Brief intro (1 line)
   1. Place Name — one sentence description
   2. Place Name — one sentence description
   (max 5 places)

   PRICES question → always show full price table, never summarize numbers.

   BOOKING/CONTACT question → show contact details in clean list format.

   GENERAL question → max 3 short sentences. No bullet points.

3. NEVER write more than 150 words total.
4. NEVER merge everything into one paragraph.
5. Always be warm but concise."""
# Memory Store
conversation_store = {}

# Context Router
def get_context(question):
    q = question.lower()

    if any(w in q for w in ["phone", "email", "contact", "book",
                              "reservation", "payment", "iban", "bank"]):
        return CONTACT_INFO, "contacts"

    elif any(w in q for w in ["weather now", "weather today", "weather in gilgit",
                           "temperature today", "temperature in gilgit",
                           "road condition", "road open", "road closed",
                           "road status", "highway open", "highway closed",
                           "highway status", "karakoram", "kkh",
                           "flight status", "flight today", "pia flight",
                           "is it raining", "raining in gilgit",
                           "road to hunza", "babusar open today"]):
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
    messages += history[-4:]  # reduced from 6 to 4
    messages.append({
        "role": "user",
        "content": f"""Guest question: {question}

Information:
{context}

Answer using the correct format for this question type. Max 150 words."""
    })

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=400,  # reduced from 1024
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
