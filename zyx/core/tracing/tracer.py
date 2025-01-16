"""
### zyx.core.tracing.tracer

Contains the Tracer class, which is a global library singleton,
as well as the zyx.trace() function.

All modules in `zyx` do is log something and assign a pattern to it,
this module matches user inputs with registered patterns. This is important
for cases such as user named agents, where a hook might look like :
`agent_43:completion:request`
The tracer will log to anything named or with the prefix 'agent_43' that
has a completion filter with the event request.
"""