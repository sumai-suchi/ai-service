import os
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.llm import llm

from app.tools.operations_tools import fetch_operations_summary

load_dotenv()



# Agent State
class OperationsAgentState(TypedDict):
    messages: List[Dict[str, Any]]
    manager_id: Optional[str]
    operations_data: Optional[Dict[str, Any]]
    attention_required_count: int


# Node 1: Fetch Real-time Operational Snapshot
async def fetch_operations_node(state: OperationsAgentState):
    manager_id = state.get("manager_id")
    data = await fetch_operations_summary(manager_id)

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
    manager_info = ops_data.get("manager") or {}
    manager_name = manager_info.get("name", "Operations Manager")
    user_query = state["messages"][-1].content

    system_prompt = (
        f"You are the Operations Dispatch Assistant AI for ParcelPilot, assisting {manager_name}.\n"
        "Your role is to assist Operations Managers with crisp, actionable logistics insights.\n\n"
        "STRICT OPERATIONAL RULES:\n"
        "1. Base your answer EXCLUSIVELY on the verified [OPERATIONS_SNAPSHOT].\n"
        "2. Address the manager politely (e.g. 'Hello Manager {manager_name}').\n"
        "3. Always group unassigned shipments and issues by Hub (e.g., Dhaka Hub, Rajshahi Hub, Chittagong Hub).\n"
        "4. Explicitly highlight urgent bottlenecks (e.g., waiting > 2 hours, repeated delivery failures, courier shortages).\n"
        "5. Provide 1-2 actionable next steps (e.g., 'Recommend reassigning 2 idle couriers from Dhaka North to Central Hub').\n"
        "6. Keep the tone professional, concise, and structured with bullet points."
    )

    prompt_content = (
        f"Manager Query: '{user_query}'\n\n"
        f"[OPERATIONS_SNAPSHOT]:\n{ops_data}\n\n"
        "Provide a direct, high-level operational answer answering the manager's query."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    response = await llm.ainvoke(messages)
    content = response.content
    ai_reply = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content) if isinstance(content, list) else str(content)

    return {"messages": [AIMessage(content=ai_reply)]}


# Build LangGraph Workflow
workflow = StateGraph(OperationsAgentState)
workflow.add_node("fetch_operations", fetch_operations_node)
workflow.add_node("generate_insights", generate_operations_insights_node)

workflow.add_edge(START, "fetch_operations")
workflow.add_edge("fetch_operations", "generate_insights")
workflow.add_edge("generate_insights", END)

operations_agent = workflow.compile()
