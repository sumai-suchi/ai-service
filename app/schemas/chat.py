from typing import Optional
from pydantic import BaseModel


class CustomerChatRequest(BaseModel):
    message: str


class CustomerChatResponse(BaseModel):
    answer: str
    tracking_number: Optional[str] = None
    status: Optional[str] = None