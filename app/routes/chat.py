from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from app.schemas.chat import CustomerChatRequest, CustomerChatResponse
from app.agents.shipment_agent import shipment_agent

router = APIRouter(prefix="/api/v1/ai", tags=["Customer Support Agent"])


@router.post("/chat", response_model=CustomerChatResponse)
async def chat_endpoint(request: CustomerChatRequest):
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "tracking_number": None,
            "shipment_data": None,
            "is_tracking_query": False,
        }

        result = await shipment_agent.ainvoke(initial_state)
        ai_reply = result["messages"][-1].content
        data = result.get("shipment_data") or {}

        return CustomerChatResponse(
            answer=ai_reply,
            tracking_number=result.get("tracking_number"),
            status=data.get("status") if isinstance(data, dict) else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
