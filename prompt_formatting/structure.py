from pydantic import BaseModel, Field
from typing import List

# Define the structure for a single choice
class Choice(BaseModel):
    id: int = Field(..., description="A unique identifier for the choice (1 to 4).")
    text: str = Field(..., description="The text of the response option.")

class UserResponseOption(BaseModel):
    id: int = Field(..., description="Unique ID for the option (1, 2, 3, or 4).")
    text: str = Field(..., description="The user-selectable response text.")

# Define the overall structure for the model's output
class QuizResponse(BaseModel):
    narrative_text: str = Field(..., description="The AI's desperate narrative text and final question.")
    response_options: List[UserResponseOption] = Field(..., max_items=4, min_items=4, description="A list of exactly four possible response options.")