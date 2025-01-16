"""
### 🔎 zyx.core.tracing

Provides the application for the tracing (verbose logging) system in zyx.
"""

# tracing toggle & hooks
from .tracer import trace, on, emit

# debug toggle helper
from .visualizer import debug

# base class for other zyx modules to inherit from (this is how modules in zyx are traced)
# they do not directly import the tracer
from .traceable import Traceable
