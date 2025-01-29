# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI(title="QA Chatbot with RAG", description="A question-answering chatbot using Retrieval-Augmented Generation application.", version="1.0")

# Model for user queries
class Query(BaseModel):
    question: str

# Load documents for the knowledge base
logging.info("Loading documents...")
loader = TextLoader("Iris.csv")  # Replace with the actual path
raw_documents = loader.load()

# Create embeddings and FAISS vector store
logging.info("Creating vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(raw_documents, embeddings)

# Set up RetrievalQA chain
logging.info("Setting up RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever()
)

# Thread pool for handling multiple requests simultaneously
executor = ThreadPoolExecutor(max_workers=5)

# Process user question using RAG
async def process_question(question: str) -> str:
    try:
        
        logging.info(f"Processing question: {question}")
        answer = qa_chain.run(question)
        return answer
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return "Sorry, I encountered an error while processing your question."

# Endpoint for chatbot
@app.post("/chat")
async def chat(query: Query):


    try:
        future = executor.submit(process_question, query.question)
        answer = future.result()
        return {"question": query.question, "answer": answer}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Optional enhancements
@app.get("/health")
async def health_check():
    return {"status": "Chatbot is running smoothly!"}

@app.get("/docs")
async def list_documents():
    return {"documents": [doc.metadata for doc in raw_documents]}

# Frontend suggestion (hosted separately or via FastAPI)
frontend_code = """<!DOCTYPE html>
<html>
<head>
    <title>Chatbot</title>
</head>
<body>
    <h1>Question-Answer Chatbot</h1>
    <input type="text" id="question" placeholder="Ask a question" />
    <button onclick="askQuestion()">Submit</button>
    <p id="answer"></p>
    
    <script>
        async function askQuestion() {
            const question = document.getElementById("question").value;
            const response = await fetch("http://127.0.0.1:8000/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question })
            });
            const data = await response.json();
            document.getElementById("answer").textContent = data.answer;
        }
    </script>
</body>
</html>"""

@app.get("/frontend")
def get_frontend():
    return frontend_code

# Run the app (use uvicorn for production deployment)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
