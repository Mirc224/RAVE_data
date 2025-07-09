from typing import List
from pydantic import BaseModel, Field

class PresentAttributes(BaseModel):
    attributes: List[str] = Field(description="List of attributes present in the ad, selected from the predefined list.")