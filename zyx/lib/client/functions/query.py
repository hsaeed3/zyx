from enum import Enum
from pydantic import BaseModel, Field, create_model
import uuid
from typing import List, Optional, Union, Literal, Any, Type
from .... import completion
from ...types import client as client_types

class EnumAgentRoles(Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    GENERATOR = "generator"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"
    CHAT = "chat"
    TOOL = "tool"
    RETRIEVER = "retriever"

class EnumWorkflowState(Enum):
    IDLE = "idle"
    CHAT = "chat"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    COMPLETING = "completing"
    USING_TOOL = "using_tool"
    RETRIEVING = "retrieving"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str

class Plan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    tasks: List[Task] = Field(default_factory=list)

class Workflow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_goal: Optional[str] = None
    current_goal: Optional[str] = None
    plan: Optional[Plan] = None
    current_task: Optional[Task] = None
    state: EnumWorkflowState = EnumWorkflowState.IDLE
    completed_tasks: List[Task] = Field(default_factory=list)
    task_queue: List[Task] = Field(default_factory=list)

class UserIntent(BaseModel):
    intent: str
    confidence: float

class QueryParams(BaseModel):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    mode: client_types.InstructorMode = "markdown_json_mode"
    max_retries: int = 3
    organization: Optional[str] = None
    max_tokens: Optional[int] = None
    run_tools: Optional[bool] = True
    tools: Optional[List[client_types.ToolType]] = None
    parallel_tool_calls: Optional[bool] = False
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto"
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    verbose: Optional[bool] = False

class TaskCompletionCheck(BaseModel):
    is_complete: bool

def classify_intent(user_input: str, params: QueryParams) -> UserIntent:
    intent_labels = ["chat", "plan", "execute", "evaluate", "generate", "reflect", "use_tool", "retrieve"]
    classification = completion(
        client="openai",
        messages=[
            {"role": "system", "content": "Classify the following text into one of the intents: " + ", ".join(intent_labels)},
            {"role": "user", "content": user_input}
        ],
        model=params.model,
        api_key=params.api_key,
        base_url=params.base_url,
        temperature=params.temperature,
        mode=params.mode,
        response_model=UserIntent
    )
    return classification

def generate_plan(goal: str, params: QueryParams) -> Plan:
    plan_response = completion(
        client="openai",
        messages=[
            {"role": "system", "content": "Generate a plan for the following goal: " + goal},
            {"role": "user", "content": goal}
        ],
        model=params.model,
        api_key=params.api_key,
        base_url=params.base_url,
        temperature=params.temperature,
        mode=params.mode,
        response_model=Plan
    )
    return plan_response

def execute_task(task: Task, params: QueryParams) -> str:
    execute_response = completion(
        client="openai",
        messages=[
            {"role": "system", "content": "Execute the following task: " + task.description},
            {"role": "user", "content": task.description}
        ],
        model=params.model,
        api_key=params.api_key,
        base_url=params.base_url,
        temperature=params.temperature,
        mode=params.mode,
    )
    return execute_response.choices[0].message.content

def check_task_completion(task: Task, result: str, params: QueryParams) -> bool:
    check_response = completion(
        client="openai",
        messages=[
            {"role": "system", "content": "Check if the following task is complete: " + task.description},
            {"role": "user", "content": result}
        ],
        model=params.model,
        api_key=params.api_key,
        base_url=params.base_url,
        temperature=params.temperature,
        mode=params.mode,
        response_model=TaskCompletionCheck
    )
    return check_response.is_complete

def query(
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    mode: client_types.InstructorMode = "markdown_json_mode",
    max_retries: int = 3,
    organization: Optional[str] = None,
    max_tokens: Optional[int] = None,
    run_tools: Optional[bool] = True,
    tools: Optional[List[client_types.ToolType]] = None,
    parallel_tool_calls: Optional[bool] = False,
    tool_choice: Optional[Literal["none", "auto", "required"]] = "auto",
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
    stream: Optional[bool] = False,
    verbose: Optional[bool] = False
) -> str:
    params = QueryParams(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        mode=mode,
        max_retries=max_retries,
        organization=organization,
        max_tokens=max_tokens,
        run_tools=run_tools,
        tools=tools,
        parallel_tool_calls=parallel_tool_calls,
        tool_choice=tool_choice,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        stream=stream,
        verbose=verbose
    )

    workflow = Workflow()
    workflow.user_goal = prompt
    workflow.state = EnumWorkflowState.PLANNING

    # Classify intent
    intent = classify_intent(prompt, params)

    if intent.intent == "plan":
        # Generate plan
        plan = generate_plan(prompt, params)
        workflow.plan = plan
        workflow.state = EnumWorkflowState.EXECUTING

        # Execute tasks
        for task in plan.tasks:
            result = execute_task(task, params)
            workflow.completed_tasks.append(task)
            workflow.state = EnumWorkflowState.EVALUATING

            # Check if task is complete
            if check_task_completion(task, result, params):
                workflow.state = EnumWorkflowState.REFLECTING

                # Reflect on the process
                reflection_response = completion(
                    client="openai",
                    messages=[
                        {"role": "system", "content": "Reflect on the current state of the workflow and suggest improvements or next steps."},
                        {"role": "user", "content": "Reflect on the current state of the workflow and suggest improvements or next steps."}
                    ],
                    model=params.model,
                    api_key=params.api_key,
                    base_url=params.base_url,
                    temperature=params.temperature,
                    mode=params.mode,
                )
                return reflection_response.choices[0].message.content
        workflow.state = EnumWorkflowState.COMPLETING
        final_summary = completion(
            client="openai",
            messages=[
                {"role": "system", "content": "Summarize the results of the executed plan."},
                {"role": "user", "content": "Summarize the results of the executed plan."}
            ],
            model=params.model,
            api_key=params.api_key,
            base_url=params.base_url,
            temperature=params.temperature,
            mode=params.mode,
        )
        return final_summary.choices[0].message.content

    else:
        # Handle other intents (chat, generate, etc.)
        response = completion(
            client="openai",
            messages=[
                {"role": "system", "content": "Handle the following user input: " + prompt},
                {"role": "user", "content": prompt}
            ],
            model=params.model,
            api_key=params.api_key,
            base_url=params.base_url,
            temperature=params.temperature,
            mode=params.mode,
        )
        return response.choices[0].message.content
    

if __name__ == "__main__":
    print(query("I want to learn how to code", verbose = True))