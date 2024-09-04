from typing import Literal, Optional, Union, Any, Dict, List, Tuple
from ..core.main import BaseModel
from .. import logger
from crewai.crew import Crew


# Init LLM
def _init_llm(
    model: Optional[str] = "openai/gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
):
    try:
        if model.startswith("ollama/") or model.startswith("ollama-chat/"):
            from langchain_openai.chat_models import ChatOpenAI

            if model.startswith("ollama/"):
                chat_model = model[7:]
            elif model.startswith("ollama-chat/"):
                chat_model = model[12:]

            llm = ChatOpenAI(
                model=chat_model,
                base_url="http://localhost:11434/v1" if base_url is None else base_url,
                api_key="ollama" if api_key is None else api_key,
            )
        elif model.startswith("openai/") or model.startswith("gpt-"):
            from langchain_openai.chat_models import ChatOpenAI

            if model.startswith("openai/"):
                model = model[7:]

            llm = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=api_key,
                organization=organization,
            )
        else:
            from langchain_community.chat_models.litellm import ChatLiteLLM

            llm = ChatLiteLLM(
                model_name=model,
                api_base=base_url,
                openai_api_key=api_key,
                organization=organization,
            )
        return llm
    except Exception as e:
        logger.critical(e)
        raise


MODES = Literal["hierarchical", "sequential"]

# Global variables to keep track of agent IDs and tasks
AGENT_IDS: List[str] = []
AGENT_TASKS: List[str] = []


# Function to update agent literals dynamically
def agent_literals() -> Tuple[str, ...]:
    return tuple(AGENT_IDS)


# Function to update task literals dynamically
def task_literals() -> Tuple[str, ...]:
    return tuple(AGENT_TASKS)


class MemoryConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    vector_dimension: int = 1024
    config: Dict = {}


class AgentConfig(BaseModel):
    role: str = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    allow_delegation: Optional[bool] = False
    cache: Optional[bool] = False
    tools: Optional[list] = None
    memory: Optional[bool] = False


class AgentParams(BaseModel):
    agents: list[Any] = None
    agent_configs: list[AgentConfig] = None

    agent_process: MODES = "hierarchical"
    agent_process_Process: Any = None

    use_memory: bool = False
    memory_config: Optional[MemoryConfig] = None

    use_supervisor: bool = False
    supervisor: Any = None

    tasks: list[Any] = None


class Agents:
    def __init__(
        self,
        model: Optional[str] = "openai/gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        process: MODES = "hierarchical",
        memory: Optional[bool] = False,
        memory_provider: Optional[str] = "openai",
        memory_model: Optional[str] = "text-embedding-3-small",
        memory_vector_dimension: Optional[int] = 1024,
        verbose: Optional[bool] = False,
    ):
        self.verbose = verbose

        # Initialize Agent Params
        self.params = AgentParams(
            agents=[],
            agent_configs=[],
            agent_process=process,
            use_memory=memory,
            memory_config=None,
            use_supervisor=False,
            supervisor=None,
            tasks=[],
        )

        self.llm = _init_llm(
            model=model, base_url=base_url, api_key=api_key, organization=organization
        )
        if self.verbose:
            logger.info(f"LLM initialized with model: {model}")

        # Select Core Process
        try:
            self.params.agent_process_Process = self._select_process(process=process)
            if self.verbose:
                logger.info(f"Selected process: {process}")
        except ValueError as e:
            logger.critical(e)

        # Build Memory if Memory
        if memory:
            try:
                self.params.memory_config = MemoryConfig(
                    provider=memory_provider,
                    model=memory_model,
                    vector_dimension=memory_vector_dimension,
                    config={
                        "provider": memory_provider,
                        "config": {
                            "model": memory_model,
                        },
                    },
                )
                if self.verbose:
                    logger.info(
                        f"Memory built with config: {self.params.memory_config}"
                    )
            except Exception as e:
                logger.critical(e)

    # Select Process
    def _select_process(
        self,
        process: MODES,
    ):
        from crewai.process import Process

        if process == "hierarchical":
            return Process.hierarchical
        elif process == "sequential":
            return Process.sequential

    # Internal Method to Add Agent
    def _add_agent(self, config: AgentConfig):
        try:
            from crewai.agent import Agent

            global AGENT_IDS
            AGENT_IDS.append(config.role)

            agent = Agent(
                role=config.role,
                goal=config.goal,
                backstory=config.backstory,
                tools=config.tools,
                memory=config.memory,
                allow_delegation=config.allow_delegation,
                cache=config.cache,
                verbose=self.verbose,
                llm=self.llm,
            )

            # Add the agent to self.params.agents
            self.params.agents.append(agent)

            if self.verbose:
                logger.info(f"Agent created: {config.role}")

            return agent

        except Exception as e:
            logger.critical(e)
            raise

    # Add Agent
    def add_agent(
        self,
        role: str,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
        tools: Optional[list] = None,
        memory: Optional[bool] = False,
        allow_delegation: Optional[bool] = False,
        cache: Optional[bool] = False,
        supervisor: Optional[bool] = False,
        return_agent: Optional[bool] = False,
    ):
        agent = self._add_agent(
            config=AgentConfig(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools or [],
                memory=memory,
                allow_delegation=allow_delegation,
                cache=cache,
            )
        )

        if supervisor:
            self.params.use_supervisor = True
            self.params.supervisor = agent
            if self.verbose:
                logger.info(f"Supervisor created: {role}")

        if return_agent:
            return agent

    # Add Task
    def add_task(
        self,
        description: str = None,
        agent: Literal[agent_literals()] = None,
        output: str = None,
        human_input: Optional[bool] = False,
        response_model: Optional[Union[BaseModel, dict]] = None,
        output_file: Optional[str] = None,
        tools: Optional[list] = None,
        context: Union[Any, List[Any]] = None,
    ):
        if description is None:
            raise ValueError("Task description cannot be None")

        task_agent = None
        task_context = []
        task_tools = tools if tools is not None else []

        if agent:
            if len(AGENT_IDS) == 0:
                logger.warning(
                    "No agents found. Please create an Agent before adding a task."
                )
            elif agent not in AGENT_IDS:
                logger.error(f"Agent {agent} not found in {AGENT_IDS}")
            else:
                agent_index = AGENT_IDS.index(agent)
                if agent_index < len(self.params.agents):
                    task_agent = self.params.agents[agent_index]
                    if self.verbose:
                        logger.info(f"Task added to agent: {task_agent.role}")
                    # Add agent's tools to task_tools
                    if hasattr(task_agent, "tools") and task_agent.tools:
                        task_tools.extend(task_agent.tools)
                else:
                    logger.error(
                        f"Agent {agent} found in AGENT_IDS but not in self.params.agents"
                    )

        if context:
            if not isinstance(context, list):
                task_context = [context]
            else:
                task_context = context

        try:
            from crewai.task import Task

            task_kwargs = {
                "description": description,
                "agent": task_agent,
                "expected_output": output,
                "human_input": human_input,
                "context": task_context,
            }

            if task_tools:
                task_kwargs["tools"] = task_tools

            if response_model:
                if isinstance(response_model, BaseModel):
                    task_kwargs["output_pydantic"] = response_model
                elif isinstance(response_model, dict):
                    task_kwargs["output_json"] = response_model

            if output_file:
                task_kwargs["output_file"] = output_file

            new_task = Task(**task_kwargs)

            self.params.tasks.append(new_task)
            AGENT_TASKS.append(description)

            if self.verbose:
                logger.info(f"Task added: {description}")

            return new_task

        except Exception as e:
            logger.error(f"Error adding task: {e}")
            raise

    # Run
    def run(
        self,
        planning: Optional[bool] = False,
        cache: Optional[bool] = False,
        return_crew: Optional[bool] = False,
    ) -> Crew:
        try:
            crew = Crew(
                tasks=self.params.tasks,
                agents=self.params.agents,
                process=self.params.agent_process_Process,
                verbose=self.verbose,
                manager_llm=self.llm
                if self.params.use_supervisor
                or self.params.agent_process == "hierarchical"
                else None,
                memory=self.params.use_memory,
                embedder=self.params.memory_config.config
                if self.params.use_memory
                else None,
                cache=cache,
                planning=planning,
            )

            if self.verbose:
                logger.info("Built Crew")

            crew.kickoff()

            if return_crew:
                return crew
        except Exception as e:
            logger.critical(e)
            raise


if __name__ == "__main__":
    agents = Agents(model="ollama/llama3.1", verbose=True, memory=True)

    agents.add_agent("logger", goal="do some logging", backstory="He is a logger")

    agents.add_task(description="do some logging", agent="logger", output="a log file")

    agents.run()
