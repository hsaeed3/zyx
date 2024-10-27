# zyx._agents.py
# zyx // hammad saeed // 2024


# zyx core util 
from ._rich import logger, console
from . import _exceptions as exceptions

# uses the zyx basemodel for easier LLM completions and modularity
from .basemodel import  Field, BaseModel

# message utils
from .resources.utils.messages import MessagesUtils

# completions base types
from .resources.types import completion_create_params as params
# model outputs
from .resources.types import model_outputs as outputs

# ext
import enum
import random
from pydantic import create_model, ConfigDict
from typing import Type, List, Union, Optional, Any, Literal, Dict


# client defined as type
# avoids circular import
# agents is a method of the client; not a client itself
Completions = Type["Completions"]
# store
Store = Type["Store"]


# ========================================================================
# EXCEPTIONS
# ========================================================================

# top level exceptions
AgentsInitializationError = exceptions.ZyxError
AddAgentError = exceptions.ZyxError
ContextError = exceptions.ZyxError

# task exceptions
TaskInitializationError = exceptions.ZyxError

# vector store specific
StoreContentError = exceptions.ZyxError

# method exceptions
CompletionError = exceptions.ZyxError
ClassificationError = exceptions.ZyxError
GenerationError = exceptions.ZyxError
DelegationError = exceptions.ZyxError
ValidationError = exceptions.ZyxError
RunError = exceptions.ZyxError
SelectionError = exceptions.ZyxError
SummarizationError = exceptions.ZyxError
PatchError = exceptions.ZyxError
ToolError = exceptions.ZyxError


# ========================================================================
# MODELS
# ========================================================================
# Base models for agents and objects


class Object(BaseModel):

    content : Any


class State(BaseModel):

    messages : Optional[List[params.Message]] = []


# Banks of emojis and colors for agent display
emoji_bank = [
    "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😇", "😉", "😊", "😋", "😎", "😍",
    "😘", "😗", "😙", "😚", "😜", "😝", "😛", "🤑", "🤗", "🤩", "🤔", "🤨", "😐", "😑", "🙄",
    "😏", "😣", "😥", "😮", "🤐", "😯", "😪", "😫", "😴", "😌", "😛", "🤓", "🤣",
]

rich_color_bank = [
    "blue3", "blue1", "dark_green",
    "dodger_blue2", "dodger_blue3",
    "deep_sky_blue4", "spring_green2",
    "spring_green3", "spring_green4",
    "cornflower_blue", "cadet_blue",
    "medium_orchid", "medium_orchid3",
    "rosy_brown", "grey69",
    "light_steel_blue", "light_steel_blue3",
    "deep_pink3", "orchid", "plum3", "tan",
    "misty_rose3", "thistle3", "plum2",
    "orange1", "orange_red1"
]


# Base completion parameters model
class AgentCompletionParams(BaseModel):

    # allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed = True)

    # ! TODO:
    #   - DSPY teleprompting
    # passed or generated on runtime
    system_prompt : Optional[str] = None

    # llm
    model : params.ChatModel = params.ZYX_DEFAULT_MODEL

    # params
    base_url : Optional[str] = None
    api_key : Optional[str] = None
    temperature : Optional[float] = None


# Agent model defining properties and capabilities
class Agent(BaseModel):

    # allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed = True)

    # role -- only required field (simple base prompt created)
    role : str

    # name
    name : Optional[str] = Field(default_factory=lambda: f"Agent_{random.randint(1, 1000)}")

    # description
    description : Optional[str] = None

    # store (vector_store) (zyx.data.vector_stores.store:Store)
    store : Optional[Any] = None

    # store search param
    top_k : int = 5

    # agent tools
    tools : Optional[Union[List[params.ToolType], str]] = None

    # agent completion params
    params : Optional[AgentCompletionParams] = None

    # for internal use
    emoji : Optional[str] = Field(default_factory=lambda: random.choice(emoji_bank))
    color : Optional[str] = Field(default_factory=lambda: random.choice(rich_color_bank))


# ========================================================================
# AGENTS
# ========================================================================
# Main Agents class for managing multiple agents


# AGENTSSSSTSTSTSTTAT!!!!!! WOOOO
class Agents:


# ========================================================================
# TASK HANDLER
# ========================================================================
# Task handler for managing object manipulation and agent tasks


    def task(
            self,
            object : Optional[BaseModel] = Object,
            messages : Optional[List[params.Message]] = None
    ):
        """Initialize a context handler"""

        # init handler
        return self._AgentsTaskHandler(self, object = object, messages = messages if messages else self.state.messages)


    # TASK HANDLER
    class _AgentsTaskHandler:


        def __init__(
                self,
                _parent : 'Agents',
                messages : Optional[List[params.Message]] = None,
                object : Optional[BaseModel] = Object
        ):
            
            try:

                # bring agents to self level in task context
                self._parent = _parent

                # messages
                self.messages = messages if messages else self._parent.state.messages

                # object (context)
                self.object = object

            except Exception as e:
                raise TaskInitializationError(f"Error initializing task handler: {e}") from e


        # enter context
        def __enter__(self):

            # update state messages
            self._parent.state.messages = self.messages

            return self


        # exit context
        def __exit__(self, exc_type, exc_val, exc_tb):

            self._parent.state.messages = self.messages

            pass


        # get object prompt
        # builds agent specific context & instruction
        def _get_object_prompt(
                self,
                agent: Optional[Union[str, Agent]] = None,
                message: str = ""
        ):
            try:

                messages = self.messages

                messages.append({"role": "user", "content": f"## OBJECT\n\n{str(self.object.model_dump() if not isinstance(self.object, type) else self.object.model_json_schema())}\n\n## INSTRUCTIONS\n\n{message}"})

                if agent is None:
                    system_prompt = (
                        "You are an intelligent object editor.\n"
                        "Your role is to help users modify and interact with objects based on their requests.\n"
                        "You will receive the current state of an object in JSON format, followed by the user's request.\n"
                        "You should understand the object's structure and properties, and respond appropriately to help users view, modify, or analyze the object's data.\n"
                        "Provide clear explanations of any changes or observations you make ONLY IF REQUESTED."
                    )
                else:
                    # Get the agent object if a string was provided
                    agent = self._parent._match_agent(agent)
                    if agent is None:
                        raise ValueError(f"Agent not found {agent}")

                    # Check if agent has params and system_prompt
                    if agent.params is None or agent.params.system_prompt is None:
                        system_prompt = (
                            f"You are a {agent.role}.\n"
                            f"Your task is to help build and modify objects based on user requests.\n"
                            "You will receive the current state of an object in JSON format, followed by the user's request.\n"
                            "You should understand the object's structure and properties, and respond appropriately.\n"
                            "Provide clear explanations of any changes or observations you make ONLY IF REQUESTED."
                        )
                    else:
                        system_prompt = (
                            f"{agent.params.system_prompt}\n\n"
                            "## INSTRUCTIONS\n"
                            "You are now in the context of building an object.\n"
                            "You will receive the current state of an object in JSON format, followed by the user's request.\n"
                            "You should understand the object's structure and properties, and respond appropriately to help users view, modify, or analyze the object's data.\n"
                            "Provide clear explanations of any changes or observations you make ONLY IF REQUESTED."
                        )

                messages = MessagesUtils.swap_system_prompt(
                    system_prompt={"role": "system", "content": system_prompt},
                    messages=messages
                )

                return messages
            
            except Exception as e:
                raise ContextError(f"Error getting object prompt: {e}") from e


        # select best tool
        def select_best_tool(
                self,
                agent : Optional[Union[str, Agent]] = None,
                tools : Optional[Union[List[params.ToolType], str]] = None,
                message : str = "",
                model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL
        ):
            
            try:

                # Get object context and add to message
                object_context = f"## OBJECT\n\n{str(self.object.model_dump() if not isinstance(self.object, type) else self.object.model_json_schema())}"
                task_message = f"For the following object:\n\n{object_context}\n\nPlease select the best tool or steps of tools to use to complete this task: {message}"

                self.messages.append(
                    {"role" : "user", "content" : task_message}
                )

                tools = tools if tools else []

                if agent is not None:
                    agent = self._parent._match_agent(agent)

                    if agent.tools is not None:
                        tools.extend(agent.tools)

                formatted_tools = []
                tool_names = []

                for tool in tools:
                    if not isinstance(tool, str):

                        formatted = params.Tool(function=tool)

                        formatted.convert(self._parent.verbose)

                        formatted_tools.append(f"{formatted.name}: {formatted.description}")
                        tool_names.append(formatted.name)
                    else:

                        formatted_tools.append(tool)
                        tool_names.append(tool)

                ToolNames = enum.Enum('ToolNames', {name: name for name in tool_names})

                Steps = create_model(
                    'Steps',
                    steps=(List[ToolNames], Field(default_factory=list))
                )

                response = self._parent.client.completion(
                    messages = [
                        {"role" : "system", "content" : "You are a tool selector for object manipulation and editing. If selecting multiple tools you always select sequentially, to complete the task in the message most efficiently."},
                        {"role" : "user", "content" : task_message},
                        {"role" : "user", "content" : f"Here are the available tools: {formatted_tools}"}
                    ],
                    model = model,
                    response_model = Steps
                )

                # convert the enum to the actual tool name
                returned_tools = []


                for step in response.steps:
                    try:
                        # Convert enum value to string to get the tool name
                        tool_name = step.value
                        returned_tools.append(tool_name)
                    except Exception as e:
                        raise ValueError(f"Tool {step} not found in tool list") from e


                if len(returned_tools) == 1:
                    return returned_tools[0]
                else:
                    return returned_tools
            
            except Exception as e:
                raise SelectionError(f"Error selecting best tool: {e}") from e
            

        def delegate(
            self,
            message: str,
            model: Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
            tools: Optional[List[params.ToolType]] = None
        ) -> List[Dict[str, str]]:
            """Delegate tasks to agents based on their roles and capabilities.

            Args:
                message: Task or goal to accomplish
                agents: List of available agents to delegate to
                model: Model to use for planning
                tools: Optional list of tools available to agents

            Returns:
                List of delegation steps with agent and instructions
            """
            try:
                agents = self._parent.agents

                self._parent.summarize(model = model, _check_len = True)

                # Add object context to delegation message
                object_context = f"## OBJECT\n\n{str(self.object.model_dump() if not isinstance(self.object, type) else self.object.model_json_schema())}"
                delegation_message = f"For the following object:\n\n{object_context}\n\nPlease delegate the following task: {message}"

                self.messages.append(
                    {"role" : "user", "content" : delegation_message}
                )

                formatted_agents = []
                agent_names = []
                formatted_tools = []
                agent_tools = {}

                # Format agents and their tools
                for agent in agents:
                    if isinstance(agent, Agent):
                        desc = f"{agent.role}"
                        if agent.description:
                            desc += f": {agent.description}"
                        formatted_agents.append(desc)
                        agent_names.append(agent.role)

                        # Store agent's tools if they have any
                        if hasattr(agent, 'tools') and agent.tools:
                            agent_tools[agent.role] = agent.tools
                    else:
                        formatted_agents.append(agent)
                        agent_names.append(agent)

                # Format global tools if provided
                if tools:
                    for tool in tools:
                        if not isinstance(tool, str):
                            formatted = params.Tool(function=tool)
                            formatted.convert()
                            formatted_tools.append(f"{formatted.name}: {formatted.description}")

                # Create agent enum
                AgentNames = enum.Enum('AgentNames', {str(name): str(name) for name in agent_names})

                # Create delegation step model
                DelegationStep = create_model(
                    'DelegationStep',
                    agent=(AgentNames, ...),
                    instruction=(str, ...)
                )

                Steps = create_model(
                    'Steps',
                    steps=(List[DelegationStep], Field(default_factory=list))
                )

                # Build context for planner
                context = [
                    f"Available Agents:\n{chr(10).join(formatted_agents)}"
                ]

                if formatted_tools:
                    context.append(f"\nAvailable Global Tools:\n{chr(10).join(formatted_tools)}")

                # Get delegation plan
                plan = self._parent.client.planner(
                    input=Steps,
                    instructions="\n".join([
                        "You are a task delegation planner for object manipulation and editing. Your job is to:",
                        "1. Break down the task into logical steps",
                        "2. Assign the most suitable agent for each step",
                        "3. Provide clear instructions for each agent",
                        "4. Consider agent capabilities and roles",
                        "5. Allow agents to be used multiple times if needed",
                        "\nContext:\n" + "\n".join(context),
                        "\n\n",
                        f"Query: {delegation_message}"
                    ]),
                    model=model,
                    steps=10
                )

                # Convert plan to delegation steps
                delegation_steps = []
                for step in plan.tasks:
                    for step in step.steps:
                        try:
                            # Extract agent and instruction from task
                            agent_name = step.agent.value
                            instruction = step.instruction

                            # Validate agent exists
                            if agent_name not in agent_names:
                                raise ValueError(f"Agent {agent_name} not found in agent list")

                            # Combine global tools with agent-specific tools
                            available_tools = []
                            if tools:
                                available_tools.extend(tools)
                            if agent_name in agent_tools:
                                available_tools.extend(agent_tools[agent_name])

                            # Select tools for this step if any are available
                            selected_tools = None
                            if available_tools:
                                selected_tools = self.select_best_tool(
                                    message=instruction,
                                    tools=available_tools,
                                    model=model
                                )

                            step_info = {
                                "agent": agent_name,
                                "instruction": instruction
                            }

                            if selected_tools:
                                step_info["tools"] = selected_tools

                            delegation_steps.append(step_info)

                        except Exception as e:
                            raise ValueError(f"Invalid delegation step: {step}") from e

                self.messages.append(
                    {"role" : "assistant", "content" : f"Here are the delegation steps: {delegation_steps}"}
                )

                return delegation_steps
            
            except Exception as e:
                raise DelegationError(f"Error delegating task: {e}") from e


        def completion(
                self,
                message : str,
                model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
                agent : Optional[Union[str, Agent]] = None,
                tools : Optional[Union[List[params.ToolType], str]] = None,
                response_model : Optional[BaseModel] = None,
        ):
            try:

                self._parent.summarize(model = model, _check_len = True)

                self._parent._current_message += 1

                messages = self.messages

                messages = self._get_object_prompt(agent = agent, message = message)

                response = self._parent.client.completion(
                    messages = messages,
                    model = model,
                    tools = tools,
                    response_model = response_model
                )

                return response
            
            except Exception as e:
                raise CompletionError(f"Error completing task: {e}") from e


        def generate(
                self,
                message : str = "",
                agent : Optional[Union[str, Agent]] = None,
                model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL
        ):
            
            try:

                self._parent.summarize(model = model, _check_len = True)

                self._parent._current_message += 1

                messages = self._get_object_prompt(agent = agent, message = message)

                response = self._parent.client.generate(
                    target = self.object,
                    instructions = str(messages)
                )

                self.object = response

                self.messages.append(
                    {"role" : "assistant", "content" : str(response.model_dump())}
                )

                return response
            
            except Exception as e:
                raise GenerationError(f"Error generating object: {e}") from e


        def validate(
                self,
                message : str,
                agent : Optional[Union[str, Agent]] = None,
                model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
                process : Literal["accuracy", "validate", "fact_check", "guardrails"] = "accuracy",
        ) -> Union[outputs.JudgmentResult, outputs.ValidationResult, outputs.FactCheckResult, outputs.GuardrailsResult]:
            
            try:

                self._parent.summarize(model = model, _check_len = True)

                self._parent._current_message += 1

                message = f"## OBJECT\n\n{str(self.object.model_dump() if not isinstance(self.object, type) else self.object.model_json_schema())}\n\n## INSTRUCTIONS\n\n{message}"

                self.messages.append({"role" : "user", "content" : f"Please validate the following object:\n{message}"})

                response = self._parent.client.validate(
                    prompt=message,
                    responses=[str(self.object.model_dump() if not isinstance(self.object, type) else self.object.model_json_schema())],
                    process=process,
                    schema=self._get_object_prompt(agent=agent, message=message),
                    model = model
                )

                if isinstance(response, outputs.JudgmentResult):
                    self.messages.append(
                        {"role" : "assistant", "content" : f"### Judgment\n{str(response.model_dump())}"}
                    )
                elif isinstance(response, outputs.ValidationResult):
                    self.messages.append(
                        {"role" : "assistant", "content" : f"### Validation\n{str(response.model_dump())}"  }
                    )
                elif isinstance(response, outputs.FactCheckResult):
                    self.messages.append(
                        {"role" : "assistant", "content" : f"### Fact Check\n{str(response.model_dump())}"}
                    )
                elif isinstance(response, outputs.GuardrailsResult):
                    self.messages.append(
                        {"role" : "assistant", "content" : f"### Guardrails\n{str(response.model_dump())}"}
                    )
                return response
            
            except Exception as e:
                raise ValidationError(f"Error validating object: {e}") from e


        def patch(
                self,
                instructions : str,
                model : Optional[params.ChatModel] = None,
                fields : Optional[List[str]] = None
        ):
            
            try:

                self._parent.summarize(model = model, _check_len = True)

                self._parent._current_message += 1

                # raise an error if its not initialized yet or if its
                # still a model type
                if self.object is None or isinstance(self.object, type):
                    raise ValueError("Object is not initialized yet.")

                instruction = f"Please patch the model with the following instructions:\n{instructions}"
                instruction += f"\nThe current state of the model is:\n{str(self.object.model_dump())}"
                instruction += f"Here are the fields i want modified: {fields}" if fields else ""

                self.messages.append({"role" : "user", "content" : instruction})

                response = self._parent.client.patch(
                    target = self.object,
                    model = model,
                    fields = fields,
                    instructions = instructions
                )

                self.object = response

                self.messages.append(
                    {"role" : "assistant", "content" : f"### Patched Object\n{str(response.model_dump())}"}
                )

                return response
            
            except Exception as e:
                raise PatchError(f"Error patching object: {e}") from e


# ========================================================================
# AGENTS
# ========================================================================


    def __enter__(self) -> "Agents":
        """Enter the context manager, initializing any necessary resources"""
        # Currently just returns self since no special initialization is needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources when exiting the context manager

        Args:
            exc_type: The type of the exception that occurred, if any
            exc_val: The instance of the exception that occurred, if any
            exc_tb: The traceback of the exception that occurred, if any
        """
        # Clean up resources
        self.agents.clear()
        self.state = State()

        # Don't suppress any exceptions
        return None


    def __init__(
        self,
        client,
        messages : Optional[List[params.Message]] = None,
        summary_length : int = 5,
        verbose : bool = True
    ):
        
        try:

            self.client = client

            self.verbose = verbose

            self.agents = []

            self.state = State() if not messages else State(messages = messages)

            self.summary_length = summary_length

            self._current_message = 0

        except Exception as e:
            raise AgentsInitializationError(f"Error initializing agents: {e}") from e


    def summarize(self, model : Optional[params.ChatModel] = None, _check_len : bool = False):

        try:

            if _check_len:
                if self._current_message <= self.summary_length:
                    return None

            else:
                instruction = [
                    {"role" : "system", "content" : "You are a world class summarizer."},
                    *self.state.messages,
                    {"role" : "user", "content" : f"Create a cohesive summary of our conversation so far."}
                ]

                response = self.client.completion(messages = instruction, model = model)

                self.state.messages = [
                    {"role" : "user", "content" : "What have we discussed so far?"},
                    {"role" : "assistant", "content" : str(response.choices[0].message.content)}
                ]

                self._current_message = 0

                return response
            
        except Exception as e:
            raise SummarizationError(f"Error summarizing conversation: {e}") from e


    def _match_agent(self, agent : Optional[Union[str, Agent]] = None) -> Agent:
        return agent if isinstance(agent, Agent) else next((a for a in self.agents if a.name == agent), None)


    def _get_store_context(
            self, agent : Union[str, Agent], query : str, top_k : int = 5
    ):
        try:

            agent = self._match_agent(agent)

            if agent.store is None:
                return None

            context = ""

            results = agent.store.search(
                query = query,
                top_k = agent.top_k if agent.top_k else top_k
            )

            for result in results.results:
                context += f"{result.text}\n"

            return context
        
        except Exception as e:
            raise StoreContentError(f"Error getting store context: {e}") from e


    def add(
            self,

            role : str,
            name : Optional[str] = None,
            description : Optional[str] = None,
            store : Optional[Store] = None,
            top_k : int = 5,
            tools : Optional[Union[List[params.ToolType], str]] = None,

            system_prompt : Optional[str] = None,

            model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
            temperature : Optional[float] = None,
            base_url : Optional[str] = None,
            api_key : Optional[str] = None,

            optimize : bool = False,
            prompt_type : Union[outputs.PROMPT_TYPES, 
                                Literal["costar", "instruction", "reasoning", "tidd-ec"]] = "costar",
            process : Optional[Literal["batch", "sequential"]] = "batch"
    ):
        
        try:

            # initialize base agent
            agent = Agent(
                role = role,
                name = name,
                description = description,
                store = store,
                top_k = top_k,
                tools = tools,
                params = AgentCompletionParams(
                    model = model,
                    temperature = temperature,
                    base_url = base_url,
                    api_key = api_key
                ),
                optimize = optimize
            )

            if system_prompt is None:

                # if optimize is none, create a default system prompt
                if not optimize:

                    system_prompt = f"You are a {role}.\n"
                    if name is not None:
                        system_prompt += f"Your name is {name}.\n"
                    if description is not None:
                        system_prompt += f"You are described as {description}.\n"

                    if tools is not None:
                        system_prompt += f"You have the following tools: {tools}\n"
                        system_prompt += f"Utilize these methods to answer questions and complete tasks."

                    agent.params.system_prompt = system_prompt

                else:

                    instruction = (
                            "You are creating a prompt to guide an LLM to act, reason & react following the given instructions."
                            "\n"
                            f"The agent's name is : {name}"
                            f"The agent's role is : {role}" if role else ""
                            f"The agent's is described as : {description}" if description else ""
                        )

                    if tools:
                        instruction += "\nThe agent has been provided with the following tools:"
                        for tool in tools:
                            instruction += f"\nName: {tool.name if hasattr(tool, 'name') else tool.__name__}\n"
                            instruction += f"Description: {tool.description if hasattr(tool, 'description') else tool.__doc__}\n"

                    system_prompt = self.client.prompter(
                        instructions = instruction,
                        type = prompt_type,
                        model = model,
                        temperature = temperature,
                        api_key = api_key,
                        base_url = base_url,
                        verbose = self.client.config.verbose,
                        process = process
                    )

                    agent.params.system_prompt = system_prompt

            else:

                agent.params.system_prompt = system_prompt

            self.agents.append(agent)

            if self.verbose:
                console.print(f"[bold {agent.color}]{agent.emoji} {agent.name}[/bold {agent.color}] Initialized.")

            return agent
        
        except Exception as e:
            raise AddAgentError(f"Error adding agent: {e}") from e
    

    def chat(
        self,
        agent : Optional[Union[str, Agent]] = None,
        model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        tools : Optional[Union[List[params.ToolType], str]] = None,
        response_model : Optional[BaseModel] = None
    ):
        while True:

            message = console.input("[bold green] > [/bold green]")

            response = self.completion(
                message = message,
                agent = agent,
                model = model,
                tools = tools,
                response_model = response_model,
                stream = False
            )

            print(f"[bold green]{response.choices[0].message.content}[/bold green]")

            console.print("\n")


    def completion(
        self,
        message : str,
        agent : Optional[Union[str, Agent]] = None,
        model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        tools : Optional[Union[List[params.ToolType], str]] = None,
        response_model : Optional[BaseModel] = None,
        stream : bool = False
    ):
        
        try:

            self.summarize(model = model, _check_len = True)

            self._current_message += 1

            # get messages
            messages = self.state.messages

            # add message
            messages.append({"role" : "user", "content" : message})

            # swap system prompt if agent is provided
            if agent is not None:

                # match agent
                agent = self._match_agent(agent)

                # swap system prompt
                messages = MessagesUtils.swap_system_prompt(
                    system_prompt = {"role" : "system", "content" : str(agent.params.system_prompt)},
                    messages = messages
                )

                if tools and agent.tools is not None:

                    tools.extend(agent.tools)

                if not tools and agent.tools is not None:
                    tools = agent.tools

            # get completion
            response = self.client.completion(
                messages = messages,
                model = model,
                tools = tools,
                response_model = response_model,
                stream = False
            )

            if not stream:

                self.state.messages.append(
                    response.choices[0].message.model_dump()
                )

            else:

                streamed_content = ""
                for chunk in response:
                    for stream in chunk.choices[0].delta.content:
                        streamed_content += stream

                self.state.messages.append(
                    {"role" : "assistant", "content" : streamed_content}
                )

                return streamed_content

            return response
        
        except Exception as e:
            raise CompletionError(f"Error completing task: {e}") from e


    def select_best_tool(
            self,
            agent : Optional[Union[str, Agent]] = None,
            tools : Optional[Union[List[params.ToolType], str]] = None,
            message : str = "",
            model : Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
            process : Literal["batch", "sequential"] = "batch"
    ):
        
        try:

            self.state.messages.append(
                {"role" : "user", "content" : f"Please select the best tool or steps of tools to use to complete the following task: {message}"}
            )

            tools = tools if tools else []

            if agent is not None:
                agent = self._match_agent(agent)

                if agent.tools is not None:
                    tools.extend(agent.tools)

            formatted_tools = []
            tool_names = []

            for tool in tools:
                if not isinstance(tool, str):

                    formatted = params.Tool(function=tool)

                    formatted.convert(self._parent.verbose)

                    formatted_tools.append(f"{formatted.name}: {formatted.description}")
                    tool_names.append(formatted.name)
                else:

                    formatted_tools.append(tool)
                    tool_names.append(tool)

            ToolNames = enum.Enum('ToolNames', {name: name for name in tool_names})

            Steps = create_model(
                'Steps',
                steps=(List[ToolNames], Field(default_factory=list))
            )

            response = self.client.completion(
                messages = [
                    {"role" : "system", "content" : "You are a tool selector. If selecting multiple tools you always select sequentially, to complete the task in the message most efficiently."},
                    {"role" : "user", "content" : f"Please select the best tool or steps of tools to use to complete the following task: {message}"},
                    {"role" : "user", "content" : f"Here are the available tools: {formatted_tools}"}
                ],
                model = model,
                response_model = Steps
            )

            # convert the enum to the actual tool name
            returned_tools = []


            for step in response.steps:
                try:
                    # Convert enum value to string to get the tool name
                    tool_name = step.value
                    returned_tools.append(tool_name)
                except Exception as e:
                    raise ValueError(f"Tool {step} not found in tool list") from e


            self.state.messages.append(
                {"role" : "assistant", "content" : f"Here are the selected tools: {returned_tools}"}
            )


            if len(returned_tools) == 1:
                return returned_tools[0]
            else:
                return returned_tools
        
        except Exception as e:
            raise SelectionError(f"Error selecting best tool: {e}") from e


    def delegate(
            self,
            message: str,
            model: Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
            tools: Optional[List[params.ToolType]] = None
        ) -> List[Dict[str, str]]:
            """Delegate tasks to agents based on their roles and capabilities.

            Args:
                message: Task or goal to accomplish
                agents: List of available agents to delegate to
                model: Model to use for planning
                tools: Optional list of tools available to agents

            Returns:
                List of delegation steps with agent and instructions
            """

            try:

                agents = self.agents

                self.summarize(model = model, _check_len = True)

                self.state.messages.append(
                    {"role" : "user", "content" : f"Please delegate the following task: {message}"}
                )

                formatted_agents = []
                agent_names = []
                formatted_tools = []
                agent_tools = {}

                # Format agents and their tools
                for agent in agents:
                    if isinstance(agent, Agent):
                        desc = f"{agent.role}"
                        if agent.description:
                            desc += f": {agent.description}"
                        formatted_agents.append(desc)
                        agent_names.append(agent.role)

                        # Store agent's tools if they have any
                        if hasattr(agent, 'tools') and agent.tools:
                            agent_tools[agent.role] = agent.tools
                    else:
                        formatted_agents.append(agent)
                        agent_names.append(agent)

                # Format global tools if provided
                if tools:
                    for tool in tools:
                        if not isinstance(tool, str):
                            formatted = params.Tool(function=tool)
                            formatted.convert(self.verbose)
                            formatted_tools.append(f"{formatted.name}: {formatted.description}")

                # Create agent enum
                AgentNames = enum.Enum('AgentNames', {str(name): str(name) for name in agent_names})

                # Create delegation step model
                DelegationStep = create_model(
                    'DelegationStep',
                    agent=(AgentNames, ...),
                    instruction=(str, ...)
                )

                Steps = create_model(
                    'Steps',
                    steps=(List[DelegationStep], Field(default_factory=list))
                )

                # Build context for planner
                context = [
                    f"Available Agents:\n{chr(10).join(formatted_agents)}"
                ]

                if formatted_tools:
                    context.append(f"\nAvailable Global Tools:\n{chr(10).join(formatted_tools)}")

                # Get delegation plan
                plan = self.client.planner(
                    input=Steps,
                    instructions="\n".join([
                        "You are a task delegation planner. Your job is to:",
                        "1. Break down the task into logical steps",
                        "2. Assign the most suitable agent for each step",
                        "3. Provide clear instructions for each agent",
                        "4. Consider agent capabilities and roles",
                        "5. Allow agents to be used multiple times if needed",
                        "\nContext:\n" + "\n".join(context),
                        "\n\n",
                        f"Query: {message}"
                    ]),
                    model=model,
                    steps=10
                )

                # Convert plan to delegation steps
                delegation_steps = []
                for step in plan.tasks:
                    for step in step.steps:
                        try:
                            # Extract agent and instruction from task
                            agent_name = step.agent.value
                            instruction = step.instruction

                            # Validate agent exists
                            if agent_name not in agent_names:
                                raise ValueError(f"Agent {agent_name} not found in agent list")

                            # Combine global tools with agent-specific tools
                            available_tools = []
                            if tools:
                                available_tools.extend(tools)
                            if agent_name in agent_tools:
                                available_tools.extend(agent_tools[agent_name])

                            # Select tools for this step if any are available
                            selected_tools = None
                            if available_tools:
                                selected_tools = self.select_tools(
                                    message=instruction,
                                    tools=available_tools,
                                    model=model
                                )

                            step_info = {
                                "agent": agent_name,
                                "instruction": instruction
                            }

                            if selected_tools:
                                step_info["tools"] = selected_tools

                            delegation_steps.append(step_info)

                        except Exception as e:
                            raise ValueError(f"Invalid delegation step: {step}") from e

                self.state.messages.append(
                    {"role" : "assistant", "content" : f"Here are the delegation steps: {delegation_steps}"}
                )

                return delegation_steps
            
            except Exception as e:
                raise DelegationError(f"Error delegating task: {e}") from e
    

    def run(
        self,
        message: str,
        model: Optional[params.ChatModel] = params.ZYX_DEFAULT_MODEL,
        object: Optional[BaseModel] = None,
        tools: Optional[List[params.ToolType]] = None
    ) -> BaseModel:
        """Run the full process: delegate, plan, select tools, execute, and patch/generate final object.

        Args:
            message: Task or goal to accomplish
            model: Model to use for planning and execution
            base_model: Optional base model to initialize the object
            tools: Optional list of tools available to agents

        Returns:
            Final object after processing
        """
        try:

            # Initialize the object
            self.object = object if object else Object()

            # Instantiate the _AgentsTaskHandler class automatically
            task_handler = self._AgentsTaskHandler(self, object=self.object)

            # Delegate tasks to agents
            delegation_steps = task_handler.delegate(
                message=message,
                model=model,
                tools=tools
            )

            # Execute each delegation step
            for step in delegation_steps:
                agent_name = step["agent"]
                instruction = step["instruction"]
                selected_tools = step.get("tools", [])

                # Execute tools if any are selected
                for tool in selected_tools:
                    task_handler.select_best_tool(
                        agent=agent_name,
                        tools=[tool],
                        message=instruction,
                        model=model
                    )

                # Patch or generate the object based on the instruction
                if "patch" in instruction.lower():
                    task_handler.patch(
                        instructions=instruction,
                        model=model
                    )
                else:
                    task_handler.generate(
                        message=instruction,
                        agent=agent_name,
                        model=model
                    )

            # Return the final object
            return self.object

        except Exception as e:
            raise RunError(f"Error running task: {e}") from e
