import os
import re
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.rag.retriever import retrieve_policy_context

from app.tools.shipment_tools import fetch_shipment_details

load_dotenv()

# Fast & stable Gemini Model
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
    policy_context: Optional[str]
    is_tracking_query: bool


# Node 1: Intent & Tracking ID Extraction
async def extract_intent_node(state: ShipmentAgentState):
    latest_message = state["messages"][-1].content
    match = TRACKING_REGEX.search(latest_message)
    tracking_num = match.group(1).upper() if match else None

    # Check if user specifically asks to track or locate their personal parcel
    is_tracking = bool(tracking_num) or any(
        kw in latest_message.lower() for kw in ["my parcel", "my order", "where is my", "track my", "kothay", "amar parcel"]
    )

    return {
        "tracking_number": tracking_num,
        "is_tracking_query": is_tracking,
    }


# Node 2: Data & Knowledge Retrieval
async def fetch_data_node(state: ShipmentAgentState):
    tracking_num = state.get("tracking_number")
    user_query = state["messages"][-1].content

    # Database lookup (if tracking number exists)
    data = await fetch_shipment_details(tracking_num) if tracking_num else None

    # RAG lookup (Search static knowledge base for relevant policies/FAQ)
    policy_info = retrieve_policy_context(user_query)

    return {
        "shipment_data": data,
        "policy_context": policy_info,
    }


# Node 3: Grounded Answer Generation
async def generate_response_node(state: ShipmentAgentState):
    shipment_data = state.get("shipment_data")
    tracking_num = state.get("tracking_number")
    policy_context = state.get("policy_context") or "No specific policy document found."
    user_query = state["messages"][-1].content

    system_prompt = (
        "You are the official Customer Support AI for ParcelPilot.\n"
        "STRICT GROUNDING RULES:\n"
        "1. For general inquiries (such as delivery timeline, shipping charges, cancellation, refund, delivery attempts, RTS, FAQs), answer directly and clearly based ONLY on [COMPANY_POLICIES].\n"
        "2. If the customer is specifically asking to track or locate their personal parcel/order (e.g., 'where is my parcel', 'my order status'), but did NOT provide a tracking number, politely ask for their tracking number (e.g., PP-20261013-273810).\n"
        "3. If a tracking number is provided, summarize the parcel's status, origin/destination hub, and latest activity based ONLY on [SHIPMENT_DATA]. If shipment is not found, politely inform them.\n"
        "4. Do NOT hallucinate rules, prices, or shipment details.\n"
        "5. Keep your tone polite, professional, and concise."
    )

    prompt_content = (
        f"Customer Query: '{user_query}'\n\n"
        f"[TRACKING_NUMBER_GIVEN]: {tracking_num or 'None'}\n"
        f"[SHIPMENT_DATA]: {shipment_data if shipment_data else 'No specific parcel tracked'}\n\n"
        f"[COMPANY_POLICIES]:\n{policy_context}\n"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    response = await llm.ainvoke(messages)

    # Handle if response content is returned as a list
    content = response.content
    if isinstance(content, list):
        ai_reply = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        ai_reply = str(content)

    return {"messages": [AIMessage(content=ai_reply)]}


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
