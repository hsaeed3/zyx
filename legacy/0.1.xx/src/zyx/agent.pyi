# zyx ==============================================================================

from langchain_core.tools import tool as tool_type
from langchain_core.prompts import ChatPromptTemplate as ChatPromptTemplate_type
from langchain_core.prompts import MessagesPlaceholder as MessagesPlaceholder_type
from langchain_core.messages import BaseMessage as BaseMessage_type
from langchain_core.messages import AIMessage as AIMessage_type
from langchain_core.messages import HumanMessage as HumanMessage_type
from langchain_core.messages import SystemMessage as SystemMessage_type
from langchain_core.messages import ToolMessage as ToolMessage_type
from langchain_core.pydantic_v1 import BaseModel as BaseModel_type
from langchain_core.pydantic_v1 import Field as Field_type
from langchain_core.runnables.config import ensure_config as ensure_config_type
from langchain_openai.chat_models.base import ChatOpenAI as ChatOpenAI_type
from langchain_openai.embeddings.base import OpenAIEmbeddings as OpenAIEmbeddings_type
from langchain_anthropic.chat_models import ChatAnthropic as ChatAnthropic_type

tool = tool_type
ChatPromptTemplate = ChatPromptTemplate_type
MessagesPlaceholder = MessagesPlaceholder_type
BaseMessage = BaseMessage_type
AIMessage = AIMessage_type
HumanMessage = HumanMessage_type
SystemMessage = SystemMessage_type
ToolMessage = ToolMessage_type
BaseModel = BaseModel_type
Field = Field_type
ensure_config = ensure_config_type
ChatOpenAI = ChatOpenAI_type
OpenAIEmbeddings = OpenAIEmbeddings_type
ChatAnthropic = ChatAnthropic_type
