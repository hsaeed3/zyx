from enum import Enum
from pydantic import BaseModel, Field, create_model
import uuid
from inspect import isclass, signature
from typing import List, Optional, Union, Literal, Any, Callable, Generator

from ...types import client as client_types
from .... import completion
from ...data.stores.vector_store import VectorStore as Memory


# ==============================
# Enumerated Models
# ==============================


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


# ==============================
# Pydantic Models
# ==============================


class MemoryModel(BaseModel):
    global_thread: List[dict[str, str]] = []
    current_context_thread: List[dict[str, str]] = []
    current_summary: List[dict[str, str]] = [
        {"role": "user", "content": "What have we talked about so far?"},
        {
            "role": "assistant",
            "content": """So far, we have recently talked about the following topics: \n
         {current_topics} \n\n
         
         Some previous topics relevant to the conversation include: \n
         {previous_topics}
         """,
        },
    ]


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str


class Plan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    tasks: List[Task] = Field(default_factory=list)


class Artifact(BaseModel):
    code: str


class Workflow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_goal: Optional[str] = None
    current_goal: Optional[str] = None
    artifacts: Optional[List[Artifact]] = None
    plan: Optional[Plan] = None
    current_task: Optional[Task] = None
    state: EnumWorkflowState = EnumWorkflowState.IDLE
    completed_tasks: List[Task] = Field(default_factory=list)
    task_queue: List[Task] = Field(default_factory=list)
    memory: MemoryModel = Field(default_factory=MemoryModel)


class AgentsParams(BaseModel):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    mode: client_types.InstructorMode = "markdown_json_mode"
    memory: Optional[Union[Literal[":memory:"], str]] = ":memory:"
    memory_collection_name: str = "agents_memory"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None


class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: EnumAgentRoles
    client_params: client_types.CompletionArgs


class UserIntent(BaseModel):
    intent: str
    confidence: float


# Simplified Response Models
class ChatResponse(BaseModel):
    content: str


class PlanResponse(BaseModel):
    goal: str
    tasks: List[str]


class ExecuteResponse(BaseModel):
    result: str


class EvaluationResponse(BaseModel):
    is_satisfactory: bool
    explanation: str


class GenerateResponse(BaseModel):
    content: str


class ReflectionResponse(BaseModel):
    reflection: str


class ToolUseResponse(BaseModel):
    tool_name: str
    result: str


class RetrieveResponse(BaseModel):
    retrieved_info: str


class YesNoResponse(BaseModel):
    answer: Literal["yes", "no"]


class ArtifactIntent(BaseModel):
    needs_change: bool


# ==============================
# Agents Framework
# ==============================


class Agents:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        mode: client_types.InstructorMode = "markdown_json_mode",
        memory_collection_name: str = "agents_memory",
        memory: Optional[Union[Literal[":memory:"], str]] = ":memory:",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
        tools: List[Callable] = None,
        retriever_tools: List[Callable] = None,
        generate_artifacts: bool = False,
    ):
        """The base class for the zyx agentic framework.

        Example:
            ```python
            from zyx import Agents

            agents = Agents(
                model="gpt-4o-mini",
                temperature=0.7,
                mode="md_json",
                memory="agents_memory",
                memory_collection_name="agents_memory",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536,
            )

            # Add tools
            agents.add_tools([tool1, tool2, tool3])

            # Add retriever tools
            agents.add_retriever_tools([retriever_tool1, retriever_tool2, retriever_tool3])

            # Generate artifacts
            agents.generate_artifacts = True

            # Start the workflow
            agents.completio("I want to build a website for my business.")
            ```

        Parameters:
            - model (str): The model to use for the agents.
            - api_key (str): The API key to use for the agents.
            - base_url (str): The base URL to use for the agents.
            - temperature (float): The temperature to use for the agents.
            - mode (str): The mode to use for the agents.
            - memory (str): The memory to use for the agents.
            - memory_collection_name (str): The memory collection name to use for the agents.
            - embedding_model (str): The embedding model to use for the agents.
            - embedding_dimensions (int): The embedding dimensions to use for the agents.
            - embedding_api_key (str): The embedding API key to use for the agents.
            - embedding_base_url (str): The embedding base URL to use for the agents.
            - verbose (bool): Whether to print verbose output.
            - debug (bool): Whether to print debug output.
            - tools (list): The tools to use for the agents.
            - retriever_tools (list): The retriever tools to use for the agents.
            - generate_artifacts (bool): Whether to generate artifacts for the agents.
        """

        from ... import logger

        self.logger = logger
        self.workflow = Workflow()
        self.verbose = verbose
        self.debug = debug
        self.params = AgentsParams(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            mode=mode,
            memory=memory,
            memory_collection_name=memory_collection_name,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
        )
        self.tools = tools or []
        self.retriever_tools = retriever_tools or []
        self.generate_artifacts = generate_artifacts
        self.artifact_thread: List[Artifact] = []

        self._build_memory()

    def _build_memory(self):
        """Builds the memory for the agents."""

        try:
            self.memory = Memory(
                collection_name=self.params.memory_collection_name,
                location=self.params.memory,
                embedding_model=self.params.embedding_model,
                embedding_dimensions=self.params.embedding_dimensions,
                embedding_api_key=self.params.embedding_api_key,
                embedding_api_base=self.params.embedding_base_url,
            )
            if self.verbose:
                print("Memory built successfully")
        except Exception as e:
            raise Exception(f"Error building memory: {e}")

    def _update_memory(self, message: dict[str, str]):
        """Updates the memory for the agents.
        
        Parameters:
            - message (dict): The message to update the memory with.
            
        Returns:
            - None
        """

        try:
            self.workflow.memory.current_context_thread.append(message)
            self.workflow.memory.global_thread.append(message)
            self.memory.add(message["content"])

            if self.generate_artifacts:
                recent_artifact = (
                    self.artifact_thread[-1] if self.artifact_thread else None
                )
                artifact_intent = self._classify_artifact_intent(
                    self.workflow.memory.current_context_thread, recent_artifact
                )
                if artifact_intent.needs_change:
                    artifact = self._generate_artifact(message["content"])
                    self.workflow.artifacts = self.workflow.artifacts or []
                    self.workflow.artifacts.append(artifact)

            if self.debug:
                self.logger.debug(f"Memory updated: {message}")
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")

    def _get_context(self) -> List[dict[str, str]]:
        """Gets the context for the agents.
        
        Returns:
            - List[dict[str, str]]: The context for the agents.
        R"""

        try:
            context = self.workflow.memory.current_context_thread[-5:]
            if not context:
                return []
            relevant_info = self.memory.search(context[-1]["content"], top_k=3)

            context_message = {
                "role": "system",
                "content": f"Relevant information from memory: {[r.text for r in relevant_info.results]}",
            }

            if self.debug:
                self.logger.debug(f"Context: {context_message}")

            if self.verbose:
                print(f"Context generated.")

        except Exception as e:
            self.logger.error(f"Error getting context: {e}")
            return []

        return [context_message] + context

    def _classify_intent(self, user_input: str) -> UserIntent:
        """Classifies the intent for the agents.
        
        Parameters:
            - user_input (str): The user input to classify.
            
        Returns:
            - UserIntent: The intent for the agents.
        P"""

        from ..functions.classify import classify

        try:
            intent_labels = [
                "chat",
                "plan",
                "execute",
                "evaluate",
                "generate",
                "reflect",
                "use_tool",
                "retrieve",
            ]
            classification = classify(
                user_input,
                intent_labels,
                model=self.params.model,
                api_key=self.params.api_key,
                base_url=self.params.base_url,
            )
            return UserIntent(
                intent=classification[0].label, confidence=0.8
            )  # Assuming confidence, as classify doesn't provide it
        except Exception as e:
            self.logger.error(f"Error classifying intent: {e}")
            return UserIntent(intent="chat", confidence=0.0)

    def _extract_goal(self, user_input: str) -> str:
        """Extracts the goal for the agents.
        
        Parameters:
            - user_input (str): The user input to extract the goal from.
            
        Returns:
            - str: The goal for the agents.
        P"""

        from ..functions.extract import extract

        class GoalModel(BaseModel):
            goal: str

        try:
            extracted = extract(
                GoalModel,
                user_input,
                model=self.params.model,
                api_key=self.params.api_key,
                base_url=self.params.base_url,
            )

            if self.debug:
                self.logger.debug(f"Extracted goal: {extracted.goal}")

            if self.verbose:
                print(f"Goal extracted.")

            return extracted.goal
        except Exception as e:
            self.logger.error(f"Error extracting goal: {e}")
            return ""

    def _should_use_tool(self, context: List[dict[str, str]]) -> bool:
        """Checks if the agents should use a tool.
        
        Parameters:
            - context (List[dict[str, str]]): The context for the agents.
            
        Returns:
            - bool: Whether the agents should use a tool.
        P"""

        try:
            if not self.tools:
                return False

            tool_query = "Based on the current context, should we use one of the available tools? Respond with 'Yes' or 'No'."
            context.append({"role": "user", "content": tool_query})

            response = completion(
                messages=context,
                model=self.params.model,
                mode=self.params.mode,
                temperature=self.params.temperature,
                api_key=self.params.api_key,
                base_url=self.params.base_url,
                response_model=YesNoResponse,
            )

            if self.debug:
                self.logger.debug(f"Tool use response: {response.answer}")

            if self.verbose:
                print(f"Tool use response generated.")

            return response.answer == "yes"

        except Exception as e:
            self.logger.error(f"Error checking if tool should be used: {e}")
            return False

    def _should_retrieve(self, context: List[dict[str, str]]) -> bool:
        """Checks if the agents should retrieve information.
        
        Parameters:
            - context (List[dict[str, str]]): The context for the agents.
            
        Returns:
            - bool: Whether the agents should retrieve information.
        P"""

        if not self.retriever_tools:
            return False

        retrieval_query = "Based on the current context, should we retrieve additional information using one of the available retriever tools? Respond with 'Yes' or 'No'."
        context.append({"role": "user", "content": retrieval_query})

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=YesNoResponse,
        )

        return response.answer == "yes"

    def _create_tool_model(self, tool: Callable) -> BaseModel:
        """Creates a Pydantic model for a callable tool. Uses the tool's signature to create the model.
        
        Parameters:
            - tool (Callable): The tool to create a model for.
            
        Returns:
            - BaseModel: The Pydantic model for the tool.
        P"""
        sig = signature(tool)
        fields = {
            param.name: (
                param.annotation if param.annotation != param.empty else Any,
                ...,
            )
            for param in sig.parameters.values()
        }
        return create_model(f"{tool.__name__}Model", **fields)

    def use_tool(self, user_input: str, context: List[dict[str, str]]) -> str:
        """Uses a tool for the agents. Executes the tool and updates the memory.
        
        Parameters:
            - user_input (str): The user input to use the tool with.
            - context (List[dict[str, str]]): The context for the agents.
            
        Returns:
            - str: The response from the agents.
        P"""

        # Find the tool to use
        class AppropriateTool(BaseModel):
            tool_name: str

        tool_system_prompt = f"""You are a tool selection agent.

        You have access to the following tools:
        {[tool.__name__ if callable(tool) else tool.__class__.__name__ for tool in self.tools]}
        """
        tool_query = f"""Based on the current context, which tool should we use? Respond with only the tool name.

        Current user input: {user_input}
        """

        tool_selection = completion(
            messages=[
                {"role": "system", "content": tool_system_prompt},
                *context,
                {"role": "user", "content": tool_query},
            ],
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=AppropriateTool,
        )

        selected_tool = next(
            (
                tool
                for tool in self.tools
                if (callable(tool) and tool.__name__ == tool_selection.tool_name)
                or (
                    isinstance(tool, BaseModel)
                    and tool.__class__.__name__ == tool_selection.tool_name
                )
            ),
            None,
        )

        if not selected_tool:
            error_message = f"No matching tool found for: {tool_selection.tool_name}"
            self._update_memory({"role": "assistant", "content": error_message})
            return error_message

        if callable(selected_tool):
            tool_model = self._create_tool_model(selected_tool)
        else:
            tool_model = selected_tool.__class__

        class ToolArgs(BaseModel):
            tool_args: tool_model

        tool_args_query = f"""Now that we've selected the {tool_selection.tool_name} tool, what arguments should we pass to it? Respond with the appropriate arguments based on the tool's requirements.

        Current user input: {user_input}
        """

        tool_args = completion(
            messages=[
                {"role": "system", "content": tool_system_prompt},
                *context,
                {"role": "user", "content": tool_args_query},
            ],
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=ToolArgs,
        )

        if callable(selected_tool):
            tool_output = selected_tool(**tool_args.tool_args.dict())
        else:
            tool_output = selected_tool(**tool_args.tool_args.dict())

        self._update_memory(
            {
                "role": "assistant",
                "content": f"Used tool: {tool_selection.tool_name} with args: {tool_args.tool_args}. Output: {tool_output}",
            }
        )
        return str(tool_output)

    def completion(
        self, user_input: str, stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Handes a message from the user.

        Example:
            ```python
            from zyx import Agents

            agents = Agents(
                model="gpt-4o-mini",
                temperature=0.7,
                mode="md_json",
                memory="agents_memory",
            )

            agents.completio("I want to build a website for my business.")
            ```

        Parameters:
            - user_input (str): The message from the user.
            - stream (bool): Whether to stream the response.

        Returns:
            - str: The response from the agents.
            - generator: A generator that yields the response from the agents.
        """

        self._update_memory({"role": "user", "content": user_input})

        # Quick path for chat if no active workflow
        if self.workflow.state == EnumWorkflowState.IDLE:
            return self._handle_chat(user_input, stream)

        intent = self._classify_intent(user_input)
        context = self._get_context()

        # Check for tool use or retrieval
        if self._should_use_tool(context):
            return self.use_tool(user_input, context)
        elif self._should_retrieve(context):
            return self.retrieve(user_input)

        # Handle different intents
        if stream:
            return self._stream_response(intent, user_input, context)
        else:
            return self._handle_intent(intent, user_input, context)

    def _handle_chat(
        self, user_input: str, stream: bool
    ) -> Union[str, Generator[str, None, None]]:
        """Handles the chat for the agents.
        
        Parameters:
            - user_input (str): The user input to handle.
            - stream (bool): Whether to stream the response.

        Returns:
            - str: The response from the agents.
            - generator: A generator that yields the response from the agents.
        """

        self.workflow.state = EnumWorkflowState.CHAT
        context = self._get_context()
        context.append({"role": "user", "content": user_input})

        if stream:
            return self._stream_chat_response(context)
        else:
            return self._generate_chat_response(context)

    def _handle_intent(
        self, intent: UserIntent, user_input: str, context: List[dict[str, str]]
    ) -> str:
        """Handles the intent for the agents.
        
        Parameters:
            - intent (UserIntent): The intent for the agents.
            - user_input (str): The user input to handle.
            - context (List[dict[str, str]]): The context for the agents.

        Returns:
            - str: The response from the agents.
        """
        if intent.intent == "chat":
            return self._generate_chat_response(context)
        elif intent.intent == "plan":
            goal = self._extract_goal(user_input)
            return self.plan(goal)
        elif intent.intent == "execute":
            return self.execute()
        elif intent.intent == "evaluate":
            return self.evaluate(user_input)
        elif intent.intent == "generate":
            return self.generate(user_input)
        elif intent.intent == "reflect":
            return self.reflect()
        else:
            return self._generate_chat_response(context)

    def _generate_chat_response(self, context: List[dict[str, str]]) -> str:
        """Generates the chat response for the agents.
        
        Parameters:
            - context (List[dict[str, str]]): The context for the agents.

        Returns:
            - str: The response from the agents.
        """

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
        )

        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        self._update_memory(assistant_message)

        self._handle_artifact_generation(context, assistant_message["content"])

        return assistant_message["content"]

    def _handle_artifact_generation(self, context: List[dict[str, str]], content: str):
        """Handles the artifact generation for the agents.
        
        Parameters:
            - context (List[dict[str, str]]): The context for the agents.
            - content (str): The content to generate an artifact from.
        """

        if self.generate_artifacts:
            recent_artifact = self.artifact_thread[-1] if self.artifact_thread else None
            artifact_intent = self._classify_artifact_intent(context, recent_artifact)
            if artifact_intent.needs_change:
                artifact = self._generate_artifact(content)
                self.workflow.artifacts = self.workflow.artifacts or []
                self.workflow.artifacts.append(artifact)

    def _stream_response(
        self, intent: UserIntent, user_input: str, context: List[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Streams the response for the agents.
        
        Parameters:
            - intent (UserIntent): The intent for the agents.
            - user_input (str): The user input to handle.
            - context (List[dict[str, str]]): The context for the agents.

        Returns:
            - generator: A generator that yields the response from the agents.
        """

        if intent.intent == "chat":
            yield from self._stream_chat_response(context)
        else:
            # For all other intents, generate a chat response based on the intent and user input
            intent_context = context + [
                {
                    "role": "system",
                    "content": f"The user's intent is: {intent.intent}. Respond accordingly.",
                },
                {"role": "user", "content": user_input},
            ]
            yield from self._stream_chat_response(intent_context)

    def _stream_chat_response(
        self, context: List[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Streams the chat response for the agents.
        
        Parameters:
            - context (List[dict[str, str]]): The context for the agents.

        Returns:
            - generator: A generator that yields the response from the agents.
        """
        from .... import completion

        full_response = ""

        for chunk in completion(
            messages=context,
            model=self.params.model,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            temperature=self.params.temperature,
            stream=True,
        ):
            yield chunk
            full_response += chunk

        assistant_message = {"role": "assistant", "content": full_response}
        self._update_memory(assistant_message)

    def chat(
        self, user_input: str, stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Chats with the agents.

        Example:
            ```python
            from zyx import Agents

            agents = Agents(
                model="gpt-4o-mini",
                temperature=0.7,
                mode="md_json",
                memory="agents_memory",
            )

            agents.chat("I want to build a website for my business.")
            ```

        Parameters:
            - user_input (str): The message from the user.
            - stream (bool): Whether to stream the response.

        Returns:
            - str: The response from the agents.
            - generator: A generator that yields the response from the agents.
        """

        self.workflow.state = EnumWorkflowState.CHAT

        context = self._get_context()
        context.append({"role": "user", "content": user_input})

        if stream:
            return self._stream_chat_response(context)
        else:
            return self._generate_chat_response(context)

    def plan(self, goal: str) -> str:
        """Plans the agents.

        Example:
            ```python
            from zyx import Agents

            agents = Agents(
                model="gpt-4o-mini",
                temperature=0.7,
                mode="md_json",
                memory="agents_memory",
            )

            agents.plan("I want to build a website for my business.")
            ```

        Parameters:
            - goal (str): The goal for the agents.

        Returns:
            - str: The plan from the agents.
        """

        from ..functions.plan import plan

        self.workflow.user_goal = goal
        self.workflow.state = EnumWorkflowState.PLANNING

        generated_plan = plan(
            goal,
            model=self.params.model,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
        )
        self.workflow.plan = generated_plan

        plan_response = PlanResponse(
            goal=goal, tasks=[task.description for task in generated_plan.tasks]
        )
        plan_summary = (
            f"Created plan for goal: {plan_response.goal}\nTasks:\n"
            + "\n".join([f"- {task}" for task in plan_response.tasks])
        )
        self._update_memory({"role": "assistant", "content": plan_summary})

        if self.generate_artifacts:
            recent_artifact = self.artifact_thread[-1] if self.artifact_thread else None
            artifact_intent = self._classify_artifact_intent(
                self.workflow.memory.current_context_thread, recent_artifact
            )
            if artifact_intent.needs_change:
                artifact = self._generate_artifact(plan_summary)
                self.workflow.artifacts = self.workflow.artifacts or []
                self.workflow.artifacts.append(artifact)

        return plan_summary

    def execute(self) -> str:
        """Executes the next task in the plan."""
        if not self.workflow.plan or not self.workflow.plan.tasks:
            return "No plan or tasks available to execute."

        self.workflow.state = EnumWorkflowState.EXECUTING
        task = self.workflow.plan.tasks[0]
        self.workflow.current_task = task

        context = self._get_context()
        context.append(
            {
                "role": "user",
                "content": f"Execute the following task: {task.description}",
            }
        )

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=ExecuteResponse,
        )

        result = response.result
        self._update_memory(
            {
                "role": "assistant",
                "content": f"Executed task: {task.description}\nResult: {result}",
            }
        )

        self.workflow.completed_tasks.append(task)
        self.workflow.plan.tasks.pop(0)

        if self.generate_artifacts:
            recent_artifact = self.artifact_thread[-1] if self.artifact_thread else None
            artifact_intent = self._classify_artifact_intent(
                self.workflow.memory.current_context_thread, recent_artifact
            )
            if artifact_intent.needs_change:
                artifact = self._generate_artifact(result)
                self.workflow.artifacts = self.workflow.artifacts or []
                self.workflow.artifacts.append(artifact)

        return result

    def evaluate(self, result: str) -> str:
        """Evaluates the result of the task.

        Parameters:
            - result (str): The result of the task.

        Returns:
            - str: The evaluation of the result.
        """

        self.workflow.state = EnumWorkflowState.EVALUATING

        context = self._get_context()
        context.append(
            {"role": "user", "content": f"Evaluate the following result: {result}"}
        )

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=EvaluationResponse,
        )

        evaluation_result = f"Evaluation: {'Satisfactory' if response.is_satisfactory else 'Unsatisfactory'}\nExplanation: {response.explanation}"
        self._update_memory({"role": "assistant", "content": evaluation_result})

        if self.generate_artifacts:
            recent_artifact = self.artifact_thread[-1] if self.artifact_thread else None
            artifact_intent = self._classify_artifact_intent(
                self.workflow.memory.current_context_thread, recent_artifact
            )
            if artifact_intent.needs_change:
                artifact = self._generate_artifact(evaluation_result)
                self.workflow.artifacts = self.workflow.artifacts or []
                self.workflow.artifacts.append(artifact)

        return evaluation_result

    def generate(self, prompt: str) -> str:
        """Generates content based on the prompt.

        Parameters:
            - prompt (str): The prompt for the agents.

        Returns:
            - str: The generated content.
        """

        self.workflow.state = EnumWorkflowState.GENERATING
        context = self._get_context()
        context.append(
            {"role": "user", "content": f"Generate content based on: {prompt}"}
        )

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=GenerateResponse,
        )

        generated_content = response.content
        self._update_memory(
            {"role": "assistant", "content": f"Generated content: {generated_content}"}
        )

        if self.generate_artifacts:
            recent_artifact = self.artifact_thread[-1] if self.artifact_thread else None
            artifact_intent = self._classify_artifact_intent(
                self.workflow.memory.current_context_thread, recent_artifact
            )
            if artifact_intent.needs_change:
                artifact = self._generate_artifact(generated_content)
                self.workflow.artifacts = self.workflow.artifacts or []
                self.workflow.artifacts.append(artifact)

        return generated_content

    def reflect(self) -> str:
        """Reflects on the current state of the workflow and suggests improvements or next steps.

        Returns:
            - str: The reflection from the agents.
        """

        self.workflow.state = EnumWorkflowState.REFLECTING

        context = self._get_context()
        context.append(
            {
                "role": "user",
                "content": "Reflect on the current state of the workflow and suggest improvements or next steps.",
            }
        )

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=ReflectionResponse,
        )

        reflection = response.reflection
        self._update_memory(
            {"role": "assistant", "content": f"Reflection: {reflection}"}
        )

        if self.generate_artifacts:
            recent_artifact = self.artifact_thread[-1] if self.artifact_thread else None
            artifact_intent = self._classify_artifact_intent(
                self.workflow.memory.current_context_thread, recent_artifact
            )
            if artifact_intent.needs_change:
                artifact = self._generate_artifact(reflection)
                self.workflow.artifacts = self.workflow.artifacts or []
                self.workflow.artifacts.append(artifact)

        return reflection

    def run(self, goal: str) -> str:
        """Runs the agents.

        Parameters:
            - goal (str): The goal for the agents.

        Returns:
            - str: The final result from the agents.
        """

        self.plan(goal)
        while self.workflow.plan.tasks:
            result = self.execute()
            evaluation = self.evaluate(result)
            if "Unsatisfactory" in evaluation:
                self.plan(f"Improve the following result: {result}")
            self.reflect()

        self.workflow.state = EnumWorkflowState.COMPLETING
        final_result = self.chat("Summarize the results of the executed plan.")

        return final_result

    def _generate_artifact(self, content: str) -> Artifact:
        """Generates an artifact based on the content.

        Parameters:
            - content (str): The content for the artifact.

        Returns:
            - Artifact: The generated artifact.
        """

        context = self._get_context()
        context.append(
            {"role": "user", "content": f"Generate an artifact based on: {content}"}
        )

        response = completion(
            messages=context,
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=GenerateResponse,
        )

        artifact = Artifact(code=response.content)
        self.artifact_thread.append(artifact)

        if self.debug:
            self.logger.debug(f"Generated artifact: {artifact}")

        if self.verbose:
            print(f"Artifact generated.")

        return artifact

    def _classify_artifact_intent(
        self, context: List[dict[str, str]], recent_artifact: Optional[Artifact]
    ) -> ArtifactIntent:
        """Classifies the intent for the artifact.

        Parameters:
            - context (List[dict[str, str]]): The context for the artifact.
            - recent_artifact (Optional[Artifact]): The most recent artifact.

        Returns:
            - ArtifactIntent: The intent for the artifact.
        """

        artifact_query = f"""Based on the current context and the most recent artifact, does the artifact need to be changed? Respond with 'Yes' or 'No'.

Recent context: {context[-1]['content']}
Most recent artifact: {recent_artifact.code if recent_artifact else 'No previous artifact'}

Respond with:"""

        response = completion(
            messages=[{"role": "user", "content": artifact_query}],
            model=self.params.model,
            mode=self.params.mode,
            temperature=self.params.temperature,
            api_key=self.params.api_key,
            base_url=self.params.base_url,
            response_model=YesNoResponse,
        )

        return ArtifactIntent(needs_change=response.answer == "yes")

    def retrieve(self, query: str) -> str:
        """Retrieves information based on the query.

        Parameters:
            - query (str): The query for the agents.

        Returns:
            - str: The retrieved information.
        """
        self.workflow.state = EnumWorkflowState.RETRIEVING

        context = self._get_context()
        context.append(
            {"role": "user", "content": f"Retrieve information for: {query}"}
        )

        if not self.retriever_tools:
            return "No retriever tools available."

        # Use the first retriever tool (you might want to implement logic to choose the appropriate tool)
        retriever_tool = self.retriever_tools[0]

        try:
            retrieved_info = retriever_tool(query)
            if isinstance(retrieved_info, dict):
                # Assuming the search_web function returns a dict with a 'results' key
                retrieved_info = "\n".join(
                    [
                        f"- {result['content']}"
                        for result in retrieved_info.get("results", [])
                    ]
                )
            elif isinstance(retrieved_info, bool) and not retrieved_info:
                retrieved_info = "Failed to retrieve information."

            response = completion(
                messages=context
                + [
                    {
                        "role": "user",
                        "content": f"Summarize this information: {retrieved_info}",
                    }
                ],
                model=self.params.model,
                mode=self.params.mode,
                temperature=self.params.temperature,
                api_key=self.params.api_key,
                base_url=self.params.base_url,
                response_model=RetrieveResponse,
            )

            summary = response.retrieved_info
            self._update_memory(
                {
                    "role": "assistant",
                    "content": f"Retrieved and summarized information: {summary}",
                }
            )

            return summary
        except Exception as e:
            error_message = f"Error during retrieval: {str(e)}"
            self._update_memory({"role": "assistant", "content": error_message})
            return error_message

    def cli(self):
        print(f"Starting CLI...")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = self.completio(user_input)
            print(f"Assistant: {response}")