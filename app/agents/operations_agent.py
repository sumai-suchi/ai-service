import os
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.tools.operations_tools import fetch_operations_summary

load_dotenv()

# Fast & stable Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


# Agent State
class OperationsAgentState(TypedDict):
    messages: List[Dict[str, Any]]
    operations_data: Optional[Dict[str, Any]]
    attention_required_count: int


# Node 1: Fetch Real-time Operational Snapshot
async def fetch_operations_node(state: OperationsAgentState):
    data = await fetch_operations_summary()
    
    total_issues = 0
    if data and "summary" in data:
        summary = data["summary"]
        total_issues = (
            summary.get("totalUnassignedOver2Hours", 0)
            + summary.get("totalFailedDeliveries24h", 0)
            + summary.get("totalDelayedInTransit", 0)
        )

    return {
        "operations_data": data,
        "attention_required_count": total_issues,
    }


# Node 2: Grounded Operational Insights Generation
async def generate_operations_insights_node(state: OperationsAgentState):
    ops_data = state.get("operations_data") or {}
    user_query = state["messages"][-1].content

    system_prompt = (
        "You are the Operations Dispatch Assistant AI for ParcelPilot.\n"
        "Your role is to assist Operations Managers with crisp, actionable logistics insights.\n\n"
        "STRICT OPERATIONAL RULES:\n"
        "1. Base your answer EXCLUSIVELY on the verified [OPERATIONS_SNAPSHOT].\n"
        "2. Always group unassigned shipments and issues by Hub (e.g., Dhaka Hub, Rajshahi Hub, Chittagong Hub).\n"
        "3. Explicitly highlight urgent bottlenecks (e.g., waiting > 2 hours, repeated delivery failures, courier shortages).\n"
        "4. Provide 1-2 actionable next steps (e.g., 'Recommend reassigning 2 idle couriers from Dhaka North to Central Hub').\n"
        "5. Keep the tone professional, concise, and structured with bullet points."
    )

    prompt_content = (
        f"Manager Query: '{user_query}'\n"
        f"[OPERATIONS_SNAPSHOT]:\n{ops_data}\n\n"
        "Provide a direct, high-level operational answer answering the manager's query."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    response = await llm.ainvoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


# Build LangGraph Workflow
workflow = StateGraph(OperationsAgentState)
workflow.add_node("fetch_operations", fetch_operations_node)
workflow.add_node("generate_insights", generate_operations_insights_node)

workflow.add_edge(START, "fetch_operations")
workflow.add_edge("fetch_operations", "generate_insights")
workflow.add_edge("generate_insights", END)

operations_agent = workflow.compile()