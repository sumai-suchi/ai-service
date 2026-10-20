import os
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("PARCELPILOT_BACKEND_URL", "http://localhost:5000")
INTERNAL_KEY = os.getenv("AI_INTERNAL_SECRET", "")


async def fetch_operations_summary(manager_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch live operational metrics from the Express backend internal API."""
    url = f"{BACKEND_URL}/api/v1/internal/operations/summary"
    params = {}
    if manager_id:
        params["managerId"] = manager_id
    headers = {
        "x-internal-service-key": INTERNAL_KEY
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            if response.status_code == 200:
                body = response.json()
                return body.get("data")
            return {"error": "BACKEND_ERROR", "status": response.status_code}
        except httpx.RequestError as exc:
            return {"error": "CONNECTION_FAILED", "details": str(exc)}
