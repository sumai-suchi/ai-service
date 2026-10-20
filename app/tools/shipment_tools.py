import os
import httpx
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("PARCELPILOT_BACKEND_URL", "http://localhost:5000")
INTERNAL_KEY = os.getenv("AI_INTERNAL_SECRET", "")


async def fetch_shipment_details(tracking_number: str) -> Optional[Dict[str, Any]]:
    """Express backend er internal API theke shipment data ane."""
    url = f"{BACKEND_URL}/api/v1/internal/shipments/{tracking_number}"
    headers = {
        "x-internal-service-key": INTERNAL_KEY
    }

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                body = response.json()
                return body.get("data")
            elif response.status_code == 404:
                return {"error": "NOT_FOUND", "message": f"Shipment '{tracking_number}' khuje pawa jayni."}
            return {"error": "BACKEND_ERROR", "status": response.status_code}
        except httpx.RequestError as exc:
            return {"error": "CONNECTION_FAILED", "details": str(exc)}