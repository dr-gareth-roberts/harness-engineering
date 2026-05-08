"""Live agent REPL debugger (#10).

Wrap any `Runner` in `DebugRunner(real_runner, ...)` to pause mid-trajectory
on a configurable breakpoint, inspect the conversation history, fire ad-hoc
tool calls, mutate the next turn, and resume — programmatically or via an
interactive REPL.

See `designs/standout.md` section 10 for the full design.
"""

from harness.debug.context import DebugContext
from harness.debug.repl import DebugRepl
from harness.debug.runner import DebugAborted, DebugRunner

__all__ = [
    "DebugAborted",
    "DebugContext",
    "DebugRepl",
    "DebugRunner",
]
