from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class OperationsChatRequest(BaseModel):
    message: str
    manager_id: Optional[str] = None


class OperationsChatResponse(BaseModel):
    answer: str
    attention_required_count: int
    data_snapshot: Optional[Dict[str, Any]] = None