"""zyx.core.processing.text.templating

Jinja2 template rendering utilities for text processing.

This module provides a simple interface for rendering Jinja2 templates from
strings or files with automatic environment caching and thread-safe file reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...._internal._exceptions import ProcessingError

if TYPE_CHECKING:
    from jinja2 import Environment
    from jinja2 import Template as JinjaTemplate

__all__ = ("render_template_string",)


_ZYX_JINJA_ENV: Environment | None = None
"""The global Jinja2 environment used by zyx for template rendering."""

_ZYX_JINJA_ENV_CACHE: dict[int, tuple[dict | None, Environment]] = {}
"""Cache of Jinja2 environments keyed by config hash."""


def _get_jinja_env(config: dict | None = None) -> Environment:
    """
    Get or create a cached Jinja2 environment with the given configuration.

    Environments are cached based on config hash for reuse. The default
    environment (config=None) uses strict undefined checking, autoescape,
    and sensible whitespace handling.

    Parameters
    ----------
    config : dict | None, optional
        Optional configuration dict to customize the Jinja2 environment.
        These kwargs are passed directly to jinja2.Environment().

    Returns
    -------
    Environment
        The Jinja2 environment for the given config.
    """
    global _ZYX_JINJA_ENV, _ZYX_JINJA_ENV_CACHE

    # Create a hashable key from config
    config_key = hash(frozenset(config.items()) if config else None)

    # Check if we have a cached environment for this config
    if config_key in _ZYX_JINJA_ENV_CACHE:
        return _ZYX_JINJA_ENV_CACHE[config_key][1]

    # Create new environment
    from jinja2 import Environment, StrictUndefined

    env_kwargs = {
        "undefined": StrictUndefined,
        "autoescape": True,
        "trim_blocks": True,
        "lstrip_blocks": True,
        "keep_trailing_newline": True,
    }

    # Apply custom config if provided
    if config:
        env_kwargs.update(config)

    env = Environment(**env_kwargs)

    # Cache the environment with its config
    _ZYX_JINJA_ENV_CACHE[config_key] = (config, env)

    # Update global default if config is None
    if config is None:
        _ZYX_JINJA_ENV = env

    return env


def render_template_string(
    template_str: str,
    context: dict | None = None,
    config: dict | None = None,
) -> str:
    """
    Render a Jinja2 template string with the given context.

    This function compiles the template string using a cached Jinja2
    environment and renders it with the provided context variables.

    Parameters
    ----------
    template_str : str
        The Jinja2 template string to render.
    context : dict | None, optional
        The context dictionary containing variables for template rendering.
        Defaults to an empty dict.
    config : dict | None, optional
        Optional configuration dict to customize the Jinja2 environment.
        If None, uses the default cached environment.

    Returns
    -------
    str
        The rendered template string.

    Raises
    ------
    ProcessingError
        If template rendering fails due to syntax errors, undefined variables,
        or other Jinja2 errors.

    Examples
    --------
        ```python
        >>> template = "Hello, {{ name }}!"
        >>> render_template_string(template, {"name": "World"})
        'Hello, World!'

        >>> template = "{% for item in items %}{{ item }}{% endfor %}"
        >>> render_template_string(template, {"items": [1, 2, 3]})
        '123'
        ```
    """
    env = _get_jinja_env(config)
    template: JinjaTemplate = env.from_string(template_str)

    try:
        return template.render(context or {})
    except Exception as e:
        raise ProcessingError(f"Template rendering failed: {e}") from e
