from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI(title="ParcelPilot AI Service")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/ai/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Replace this placeholder with your LangGraph agent.
    return {
        "answer": (
            f"ParcelPilot AI received your question: "
            f"{request.message}"
        )
    }