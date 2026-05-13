"""Live agent REPL debugger (#10).

Wrap any `Runner` in `DebugRunner(real_runner, ...)` to pause mid-trajectory
on a configurable breakpoint, inspect the conversation history, fire ad-hoc
tool calls, mutate the next turn, and resume — programmatically, via an
interactive REPL, or via the Debug Adapter Protocol (so editors like
VS Code can drive the same debug session).

See `designs/standout.md` section 10 for the full design.
"""

from harness.debug.context import DebugContext
from harness.debug.dap import DapAdapter
from harness.debug.dap_protocol import DapProtocolError
from harness.debug.repl import DebugRepl
from harness.debug.runner import DebugAborted, DebugRunner

__all__ = [
    "DapAdapter",
    "DapProtocolError",
    "DebugAborted",
    "DebugContext",
    "DebugRepl",
    "DebugRunner",
]
