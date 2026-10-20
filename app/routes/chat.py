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
        raw_content = result["messages"][-1].content
        if isinstance(raw_content, list):
            ai_reply = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_content
            )
        else:
            ai_reply = str(raw_content)
        data = result.get("shipment_data") or {}

        return CustomerChatResponse(
            answer=ai_reply,
            tracking_number=result.get("tracking_number"),
            status=data.get("status") if isinstance(data, dict) else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
