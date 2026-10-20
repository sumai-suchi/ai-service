import os
import re
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.tools.shipment_tools import fetch_shipment_details

load_dotenv()

# LLM মডেল সেটআপ (fast & efficient)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

TRACKING_REGEX = re.compile(r"\b(PP-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b", re.IGNORECASE)


# Agent State
class ShipmentAgentState(TypedDict):
    messages: List[Dict[str, Any]]
    tracking_number: Optional[str]
    shipment_data: Optional[Dict[str, Any]]
    is_tracking_query: bool


# Node 1: Intent & Tracking ID Extraction
async def extract_intent_node(state: ShipmentAgentState):
    latest_message = state["messages"][-1].content
    match = TRACKING_REGEX.search(latest_message)
    tracking_num = match.group(1).upper() if match else None

    is_tracking = bool(tracking_num) or any(
        kw in latest_message.lower() for kw in ["parcel", "track", "where", "status", "delivery", "kothay"]
    )

    return {
        "tracking_number": tracking_num,
        "is_tracking_query": is_tracking,
    }


# Node 2: Database Fetch via Tool
async def fetch_data_node(state: ShipmentAgentState):
    tracking_num = state.get("tracking_number")
    if not tracking_num:
        return {"shipment_data": None}

    data = await fetch_shipment_details(tracking_num)
    return {"shipment_data": data}


# Node 3: Grounded Answer Generation
async def generate_response_node(state: ShipmentAgentState):
    shipment_data = state.get("shipment_data")
    tracking_num = state.get("tracking_number")
    user_query = state["messages"][-1].content

    system_prompt = (
        "You are the official Customer Support AI for ParcelPilot.\n"
        "STRICT RULES:\n"
        "1. Answer based ONLY on the verified [SHIPMENT_DATA] provided.\n"
        "2. Do NOT hallucinate or assume hub names, dates, or delivery statuses.\n"
        "3. If shipment data is not found, politely tell the customer the tracking number wasn't found in our system.\n"
        "4. Keep your answer helpful, natural, and concise."
    )

    if not tracking_num and state.get("is_tracking_query"):
        prompt_content = (
            f"User asked: '{user_query}'\n"
            "Context: The customer is asking about a parcel but did NOT mention any tracking number (e.g., PP-10293).\n"
            "Politely ask them to provide their ParcelPilot tracking number."
        )
    elif shipment_data and "error" in shipment_data:
        prompt_content = (
            f"User asked: '{user_query}'\n"
            f"Context: Tracking number {tracking_num} was not found in ParcelPilot.\n"
            "Inform the customer politely that this tracking number does not exist."
        )
    else:
        prompt_content = (
            f"User asked: '{user_query}'\n"
            f"[SHIPMENT_DATA]: {shipment_data}\n"
            "Summarize the parcel's current status, origin/destination hub, and latest activity clearly."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    response = await llm.ainvoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


# Build Workflow
workflow = StateGraph(ShipmentAgentState)
workflow.add_node("extract_intent", extract_intent_node)
workflow.add_node("fetch_data", fetch_data_node)
workflow.add_node("generate_response", generate_response_node)

workflow.add_edge(START, "extract_intent")
workflow.add_edge("extract_intent", "fetch_data")
workflow.add_edge("fetch_data", "generate_response")
workflow.add_edge("generate_response", END)

shipment_agent = workflow.compile()