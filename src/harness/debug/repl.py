from __future__ import annotations

import json
import shlex
import sys
from typing import TYPE_CHECKING, Any, TextIO

from harness.prompts.messages import Message, text

if TYPE_CHECKING:
    from harness.debug.context import DebugContext


_HELP_TEXT = """\
debug commands:
  help                            show this message
  messages                        list the full conversation history
  last_call                       show the most recent tool_use, if any
  turn_index                      show the count of assistant turns so far
  fire <tool> <json-args>         dispatch an ad-hoc tool call
  mutate <role> <text...>         replace the next assistant turn
  inspect <python-expr>           evaluate a Python expression with `ctx` bound
  resume                          continue execution
  abort                           terminate the run with DebugAborted
"""


class DebugRepl:
    """A small interactive REPL driven by stdin/stdout.

    Built on a `cmd.Cmd`-flavored line-loop rather than `code.InteractiveConsole`
    so behaviour is easy to script in tests by pre-populating stdin with a
    `StringIO` and capturing stdout.

    The REPL only mutates the supplied `DebugContext`; it does not return a
    value. Once the user issues `resume` or `abort` the loop exits and
    control returns to `DebugRunner`, which inspects the context to decide
    what to do next.
    """

    def __init__(
        self,
        ctx: DebugContext,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self._ctx = ctx
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout

    async def run(self) -> None:
        self._print(
            f"[harness-debug] paused at turn {self._ctx.turn_index} (type 'help' for commands)"
        )
        while True:
            self._stdout.write("> ")
            self._stdout.flush()
            line = self._stdin.readline()
            if not line:
                # EOF — treat like `resume` so a closed stdin doesn't hang.
                self._print("[harness-debug] EOF on stdin, resuming")
                self._ctx.resume()
                return
            cmd = line.strip()
            if not cmd:
                continue
            should_exit = await self._dispatch(cmd)
            if should_exit:
                return

    # ---------------------------------------------------------------- dispatch

    async def _dispatch(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self._print(f"[harness-debug] parse error: {exc}")
            return False
        if not parts:
            return False
        verb, *rest = parts

        if verb in ("help", "?"):
            self._print(_HELP_TEXT.rstrip())
            return False
        if verb == "messages":
            self._show_messages()
            return False
        if verb == "last_call":
            self._show_last_call()
            return False
        if verb == "turn_index":
            self._print(str(self._ctx.turn_index))
            return False
        if verb == "fire":
            await self._do_fire(rest)
            return False
        if verb == "mutate":
            self._do_mutate(rest)
            return False
        if verb == "inspect":
            self._do_inspect(rest)
            return False
        if verb == "resume":
            self._ctx.resume()
            self._print("[harness-debug] resuming")
            return True
        if verb == "abort":
            self._ctx.abort()
            self._print("[harness-debug] aborting")
            return True

        self._print(f"[harness-debug] unknown command: {verb!r} (try 'help')")
        return False

    # ---------------------------------------------------------------- per-command

    def _show_messages(self) -> None:
        if not self._ctx.messages:
            self._print("(no messages)")
            return
        for i, msg in enumerate(self._ctx.messages):
            summary = _summarize_message(msg)
            self._print(f"  [{i}] {msg.role}: {summary}")

    def _show_last_call(self) -> None:
        call = self._ctx.last_call
        if call is None:
            self._print("(no tool calls)")
            return
        self._print(f"tool_use({call.name}, {call.arguments})")

    async def _do_fire(self, rest: list[str]) -> None:
        if len(rest) < 1:
            self._print("usage: fire <tool> [<json-args>]")
            return
        tool = rest[0]
        raw_args = " ".join(rest[1:]).strip() or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            self._print(f"[harness-debug] arg parse error: {exc}")
            return
        if not isinstance(args, dict):
            self._print("[harness-debug] fire args must be a JSON object")
            return
        try:
            result = await self._ctx.fire(tool, args)
        except RuntimeError as exc:
            self._print(f"[harness-debug] fire failed: {exc}")
            return
        self._print(f"-> is_error={result.is_error} content={result.content!r}")

    def _do_mutate(self, rest: list[str]) -> None:
        if len(rest) < 2:
            self._print("usage: mutate <role> <text...>")
            return
        role = rest[0]
        body = " ".join(rest[1:])
        if role not in ("user", "assistant", "system"):
            self._print(f"[harness-debug] invalid role: {role!r}")
            return
        msg: Message = text(role, body)  # type: ignore[arg-type]
        self._ctx.mutate(msg)
        self._print(f"[harness-debug] queued mutation as {role}")

    def _do_inspect(self, rest: list[str]) -> None:
        if not rest:
            self._print("usage: inspect <python-expr>")
            return
        expr = " ".join(rest)
        try:
            value = evaluate_in_context(expr, self._ctx)
        except Exception as exc:  # noqa: BLE001 - surface anything to the user
            self._print(f"[harness-debug] inspect error: {exc}")
            return
        self._print(repr(value))

    # ---------------------------------------------------------------- io

    def _print(self, line: str) -> None:
        self._stdout.write(line + "\n")
        self._stdout.flush()


def evaluate_in_context(expression: str, ctx: DebugContext) -> Any:
    """Evaluate `expression` against a paused `DebugContext`.

    The DAP adapter's `evaluate` request routes through here when
    `allowEvaluate: true` is in the launch arguments (Wave 13b #17),
    sharing the same code path as the REPL's `inspect` command. This
    is the canonical "operator evaluates arbitrary Python against the
    paused state" surface; both consumers (REPL, DAP) are opt-in.

    Restricted call paths only — the function is never reached on a
    production / non-debug path. The whole point of an interactive
    debugger is letting the operator probe state with arbitrary
    expressions; restricting this would defeat the feature.
    """
    return eval(expression, {"ctx": ctx})  # noqa: S307 - debugger by design


def _summarize_message(msg: Message) -> str:
    """One-line summary suitable for the `messages` command."""
    parts: list[str] = []
    for block in msg.content:
        if block.type == "text" and block.text is not None:
            parts.append(block.text)
        elif block.type == "tool_use" and block.tool_use is not None:
            parts.append(f"tool_use({block.tool_use.name}, {block.tool_use.arguments})")
        elif block.type == "tool_result" and block.tool_result is not None:
            parts.append(f"tool_result({block.tool_result.content!r})")
        elif block.type == "file":
            parts.append(f"file({block.path})")
    return " | ".join(parts) if parts else "(empty)"
