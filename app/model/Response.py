from pydantic import BaseModel
from typing import Any, Optional


class ResponseContent(BaseModel):
    error_code: int
    message: str
    data: Optional[Any] = None