"""zyx.core.aliases

Re-exports of various components and types from the ``pydantic_ai`` library for more consistent
naming and usage throughout ``zyx``.
"""

from __future__ import annotations

from pydantic_ai import (
    Agent as PydanticAIAgent,
    RunContext as PydanticAIRunContext,
)
from pydantic_ai._instructions import (
    AgentInstructions as PydanticAIAgentInstructions,
)
from pydantic_ai.builtin_tools import (
    AbstractBuiltinTool as PydanticAIBuiltinTool,
)
from pydantic_ai.capabilities import (
    AbstractCapability as PydanticAICapability,
    CapabilityOrdering as PydanticAICapabilityOrdering,
    CapabilityPosition as PydanticAICapabilityPosition,
)
from pydantic_ai.embeddings import (
    Embedder as PydanticAIEmbedder,
    EmbeddingSettings as PydanticAIEmbeddingSettings,
    EmbeddingResult as PydanticAIEmbeddingResult,
)
from pydantic_ai.messages import (
    ModelMessage as PydanticAIMessage,
    ModelRequest as PydanticAIModelRequest,
    ModelRequestPart as PydanticAIModelRequestPart,
    ModelResponse as PydanticAIModelResponse,
    ModelResponsePart as PydanticAIModelResponsePart,
    MultiModalContent as PydanticAIMultiModalContent,
    SystemPromptPart as PydanticAISystemPromptPart,
    TextPart as PydanticAITextPart,
    ToolCallPart as PydanticAIToolCallPart,
    ToolReturnPart as PydanticAIToolReturnPart,
    UserPromptPart as PydanticAIUserPromptPart,
    UserContent as PydanticAIUserContent,
    BinaryContent as PydanticAIBinaryContent,
    ImageUrl as PydanticAIImageUrl,
    AudioUrl as PydanticAIAudioUrl,
    VideoUrl as PydanticAIVideoUrl,
    DocumentUrl as PydanticAIDocumentUrl,
)
from pydantic_ai.models import (
    Model as PydanticAIModel,
    KnownModelName as PydanticAIKnownModelName,
    ModelRequestContext as PydanticAIModelRequestContext,
)
from pydantic_ai.output import (
    NativeOutput as PydanticAINativeOutput,
    ToolOutput as PydanticAIToolOutput,
    PromptedOutput as PydanticAIPromptedOutput,
)
from pydantic_ai.result import (
    StreamedRunResult as PydanticAIAgentRunStream,
    StreamedRunResultSync as PydanticAIAgentRunStreamSync,
)
from pydantic_ai.run import AgentRunResult as PydanticAIAgentRunResult
from pydantic_ai.settings import ModelSettings as PydanticAIModelSettings
from pydantic_ai.tools import Tool as PydanticAITool
from pydantic_ai.toolsets import (
    AbstractToolset as PydanticAIToolset,
    FunctionToolset as PydanticAIFunctionToolset,
)
from pydantic_ai.usage import (
    UsageLimits as PydanticAIUsageLimits,
    RunUsage as PydanticAIRunUsage,
)

__all__ = (
    "PydanticAIAgent",
    "PydanticAIRunContext",
    "PydanticAIAgentInstructions",
    "PydanticAIBuiltinTool",
    "PydanticAICapability",
    "PydanticAICapabilityOrdering",
    "PydanticAICapabilityPosition",
    "PydanticAIEmbedder",
    "PydanticAIEmbeddingSettings",
    "PydanticAIEmbeddingResult",
    "PydanticAIMessage",
    "PydanticAIModelRequest",
    "PydanticAIModelRequestPart",
    "PydanticAIModelResponse",
    "PydanticAIModelResponsePart",
    "PydanticAIMultiModalContent",
    "PydanticAISystemPromptPart",
    "PydanticAITextPart",
    "PydanticAIToolCallPart",
    "PydanticAIToolReturnPart",
    "PydanticAIUserPromptPart",
    "PydanticAIUserContent",
    "PydanticAIBinaryContent",
    "PydanticAIImageUrl",
    "PydanticAIAudioUrl",
    "PydanticAIVideoUrl",
    "PydanticAIDocumentUrl",
    "PydanticAIModel",
    "PydanticAIKnownModelName",
    "PydanticAIModelRequestContext",
    "PydanticAINativeOutput",
    "PydanticAIToolOutput",
    "PydanticAIPromptedOutput",
    "PydanticAIAgentRunStream",
    "PydanticAIAgentRunStreamSync",
    "PydanticAIAgentRunResult",
    "PydanticAIModelSettings",
    "PydanticAITool",
    "PydanticAIToolset",
    "PydanticAIFunctionToolset",
    "PydanticAIUsageLimits",
    "PydanticAIRunUsage",
)
