from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any


PROMPT_TYPES = Literal["costar", "tidd-ec", "instruction", "reasoning"]


# -- base .completion() outputs (using types as response_model)
class StringResponse(BaseModel):
    response : str

class IntResponse(BaseModel):
    response : int

class FloatResponse(BaseModel):
    response : float

class BoolResponse(BaseModel):
    response : bool

class ListResponse(BaseModel):
    response : List[Any]


# -- for .planner()
class Task(BaseModel):
    description: str
    details: Optional[str] = None

class Plan(BaseModel):
    tasks: List[Task]


# -- for .qa()
class Question(BaseModel):
    question: str
    answer: str

class Dataset(BaseModel):
    questions: List[Question]

# -- for .solver()
class Thought(BaseModel):
    content: str
    score: float

class Thoughts(BaseModel):
    thoughts: List[Thought]

class HighLevelConcept(BaseModel):
    concept: str

class FinalAnswer(BaseModel):
    answer: str

class TreeNode(BaseModel):
    thought: Thought
    children: List["TreeNode"] = []

class TreeOfThoughtResult(BaseModel):
    final_answer: str
    reasoning_tree: TreeNode

# -- for .validator()
class JudgmentResult(BaseModel):
    explanation: str
    verdict: str

class ValidationResult(BaseModel):
    is_valid: bool
    explanation: str

class RegeneratedResponse(BaseModel):
    response: str

class FactCheckResult(BaseModel):
    is_accurate: bool
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class GuardrailsResult(BaseModel):
    passed: bool
    explanation: str

# -- for .selector()
class SelectionResult(BaseModel):
    text: str
    selected: str
    confidence: float


class MultiSelectionResult(BaseModel):
    text: str
    selections: List[str]
    confidences: List[float]