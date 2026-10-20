from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from app.schemas.operations import OperationsChatRequest, OperationsChatResponse
from app.agents.operations_agent import operations_agent

router = APIRouter(prefix="/api/v1/ai/operations", tags=["Operations Assistant Agent"])


@router.post("/chat", response_model=OperationsChatResponse)
async def operations_chat_endpoint(request: OperationsChatRequest):
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "operations_data": None,
            "attention_required_count": 0,
        }

        result = await operations_agent.ainvoke(initial_state)
        ai_reply = result["messages"][-1].content
        ops_data = result.get("operations_data") or {}

        return OperationsChatResponse(
            answer=ai_reply,
            attention_required_count=result.get("attention_required_count", 0),
            data_snapshot=ops_data.get("summary"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))