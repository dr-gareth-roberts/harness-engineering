"""DAP adapter — bridges the Debug Adapter Protocol to `DebugContext`.

`DapAdapter` runs a concurrent DAP message loop (`serve`) and a
DebugRunner-driven session simultaneously. When a breakpoint fires,
`breakpoint_callback` parks on an `asyncio.Event` until `continue` (or
`disconnect`) arrives; meanwhile `serve` keeps reading the next
inspect-style request (`variables`, `scopes`, `stackTrace`, `threads`,
`source`, limited `evaluate`) and answering it from the held
`DebugContext`. This is the load-bearing concurrency property — a
sequential implementation would deadlock on every inspect.

Wiring (typically from `harness.debug.cli`):

    adapter = DapAdapter()
    debug = DebugRunner(
        replay,
        break_on=adapter.break_on_predicate,
        breakpoint_callback=adapter.breakpoint_callback,
        dispatcher=dispatcher,
    )
    adapter.run_session = lambda: _drive(orchestrator, record)
    adapter.synthesize_source = lambda: _trajectory_lines(record)
    await adapter.serve(reader, writer)

DAP subset implemented:

- Requests: initialize, launch, setBreakpoints, configurationDone,
  threads, stackTrace, scopes, variables, evaluate (limited), source,
  continue, next, stepIn, stepOut, pause, terminate, disconnect.
- Events: initialized, stopped, continued, output, terminated, exited.

`next`, `stepIn`, `stepOut`, `pause` are accepted but treated as
"continue" — agent trajectories don't have a meaningful intra-turn
step granularity. The handlers exist so editors that rely on these
capabilities don't error out.

`evaluate` is limited to looking up the same fields the `variables`
view exposes (`turn_index`, `message_count`, `last_call.name`,
`last_call.arguments`, `pending_mutation.role`). Arbitrary-expression
evaluation is intentionally out of scope for this adapter; the REPL
(`harness debug` interactive mode) is the surface for that.

Synthetic source: `synthesize_source` (caller-supplied) returns one
line per assistant turn. DAP line N (1-based) maps to `turn_index ==
N - 1`. Setting a breakpoint at line 5 makes the runner stop right
before producing the 5th assistant turn.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from harness.debug.dap_messages import (
    Breakpoint,
    Capabilities,
    Scope,
    Source,
    StackFrame,
    Variable,
)
from harness.debug.dap_protocol import DapProtocolError, read_message, write_message

if TYPE_CHECKING:
    from harness.debug.context import DebugContext

BreakpointCallback = Callable[["DebugContext"], Awaitable[None]]
BreakPredicate = Callable[["DebugContext"], bool]
SourceProvider = Callable[[], list[str]]
SessionRunner = Callable[[], Awaitable[None]]


class DapAdapter:
    """DAP server that bridges editor traffic to a `DebugContext`."""

    THREAD_ID = 1
    """Single virtual thread for the trajectory. DAP requires at least one."""

    SOURCE_REFERENCE = 1
    """Stable handle for the synthesized trajectory source. Any positive
    int works; 1 is the simplest."""

    SCOPE_REFERENCE = 100
    """Variables reference for the single 'context' scope. Kept distinct
    from any future per-variable refs by living in its own range."""

    def __init__(self) -> None:
        # Breakpoint coordination — see module docstring on concurrency.
        self._continue_event = asyncio.Event()
        self._current_ctx: DebugContext | None = None

        # Set of 0-based turn_index values where the editor wants to break.
        self._breakpoint_turns: set[int] = set()

        # Outgoing message sequence. DAP requires a strictly increasing
        # `seq` on every adapter→editor message.
        self._out_seq = 0

        # Transport, set in `serve`.
        self._writer: asyncio.StreamWriter | None = None

        # Session state.
        self._session_task: asyncio.Task[None] | None = None
        self._launched = False
        self._terminated = False

        # Caller-supplied wiring.
        self.run_session: SessionRunner | None = None
        self.synthesize_source: SourceProvider | None = None

    # ------------------------------------------------------------------ wiring

    @property
    def break_on_predicate(self) -> BreakPredicate:
        """Closure that consults `_breakpoint_turns`. Pass to `DebugRunner`'s
        `break_on=` so `setBreakpoints` can mutate the breakpoint set
        without rebuilding the runner.
        """

        def _break(ctx: DebugContext) -> bool:
            return ctx.turn_index in self._breakpoint_turns

        return _break

    @property
    def breakpoint_callback(self) -> BreakpointCallback:
        """The async callback `DebugRunner` should fire when a breakpoint
        hits. Parks until the editor sends `continue` or `disconnect`.
        """
        return self._on_breakpoint

    # ------------------------------------------------------------------ serve

    async def serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Run the DAP message loop until disconnect, EOF, or a fatal
        protocol error. Cancels any in-flight session task on exit so
        the orchestrator doesn't leak past the editor disconnect.
        """
        self._writer = writer
        try:
            while True:
                try:
                    msg = await read_message(reader)
                except EOFError:
                    return
                except DapProtocolError as exc:
                    # Best-effort warning, then bail — the editor's state
                    # is unrecoverable once a frame goes wrong. If the
                    # warning fails to flush (broken pipe), nothing we
                    # can do about it.
                    with contextlib.suppress(Exception):
                        await self._send_output(f"protocol error: {exc}", category="stderr")
                    return

                await self._handle(msg)
                if self._terminated:
                    return
        finally:
            await self._cleanup_session()

    # ------------------------------------------------------------------ dispatch

    async def _handle(self, msg: dict[str, Any]) -> None:
        if msg.get("type") != "request":
            # Per spec, the editor only sends requests.
            return

        command = str(msg.get("command", ""))
        seq = int(msg.get("seq", 0))
        args = msg.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}

        handler_name = f"_on_{command}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            await self._respond(seq, command, success=False, message=f"unknown command: {command}")
            return

        try:
            await handler(seq, args)
        except Exception as exc:  # noqa: BLE001 - any handler error → response, not crash
            await self._respond(seq, command, success=False, message=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ commands

    async def _on_initialize(self, seq: int, args: dict[str, Any]) -> None:
        capabilities = Capabilities()
        await self._respond(seq, "initialize", body=capabilities.model_dump(by_alias=True))
        # `initialized` event tells the editor it can send breakpoints +
        # configurationDone. Per spec, this MUST come *after* the
        # initialize response.
        await self._send_event("initialized")

    async def _on_setBreakpoints(self, seq: int, args: dict[str, Any]) -> None:
        bps_in = args.get("breakpoints") or []
        verified: list[Breakpoint] = []
        new_turns: set[int] = set()

        max_line = self._max_line()
        for bp in bps_in:
            line = int(bp.get("line", 0))
            if line < 1 or line > max_line:
                verified.append(
                    Breakpoint(
                        verified=False,
                        line=line,
                        message=f"line {line} out of trajectory range (1..{max_line})",
                    )
                )
                continue
            new_turns.add(line - 1)
            verified.append(Breakpoint(verified=True, line=line))

        self._breakpoint_turns = new_turns
        await self._respond(
            seq,
            "setBreakpoints",
            body={"breakpoints": [bp.model_dump(by_alias=True) for bp in verified]},
        )

    async def _on_configurationDone(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(seq, "configurationDone")

    async def _on_launch(self, seq: int, args: dict[str, Any]) -> None:
        if self.run_session is None:
            await self._respond(
                seq, "launch", success=False, message="adapter has no run_session configured"
            )
            return
        if self._launched:
            await self._respond(seq, "launch", success=False, message="already launched")
            return
        self._launched = True
        await self._respond(seq, "launch")
        # Run the session concurrently. The DAP message loop continues
        # to pump while the orchestrator runs; both share the same event
        # loop so there's no thread coordination needed.
        self._session_task = asyncio.create_task(self._run_session_with_lifecycle())

    async def _on_continue(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(
            seq,
            "continue",
            body={"allThreadsContinued": True},
        )
        await self._resume_breakpoint()

    async def _on_next(self, seq: int, args: dict[str, Any]) -> None:
        # No intra-turn granularity yet — same as continue. Kept distinct
        # so editors that prefer step-over over continue don't get
        # silently surprised by the alias.
        await self._respond(seq, "next")
        await self._resume_breakpoint()

    async def _on_stepIn(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(seq, "stepIn")
        await self._resume_breakpoint()

    async def _on_stepOut(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(seq, "stepOut")
        await self._resume_breakpoint()

    async def _on_pause(self, seq: int, args: dict[str, Any]) -> None:
        # Pause-on-demand isn't supported — the runner only pauses at
        # configured breakpoints. Acknowledge the request so editors
        # don't error, but do nothing.
        await self._respond(seq, "pause")

    async def _on_threads(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(
            seq,
            "threads",
            body={"threads": [{"id": self.THREAD_ID, "name": "trajectory"}]},
        )

    async def _on_stackTrace(self, seq: int, args: dict[str, Any]) -> None:
        ctx = self._current_ctx
        if ctx is None:
            # No frames when not stopped.
            await self._respond(seq, "stackTrace", body={"stackFrames": [], "totalFrames": 0})
            return

        line = ctx.turn_index + 1  # DAP lines are 1-based
        frame = StackFrame(
            id=1,
            name=f"turn {ctx.turn_index}",
            line=line,
            source=Source(
                name="trajectory",
                source_reference=self.SOURCE_REFERENCE,
            ),
        )
        await self._respond(
            seq,
            "stackTrace",
            body={
                "stackFrames": [frame.model_dump(by_alias=True)],
                "totalFrames": 1,
            },
        )

    async def _on_scopes(self, seq: int, args: dict[str, Any]) -> None:
        scope = Scope(
            name="context",
            variables_reference=self.SCOPE_REFERENCE,
        )
        await self._respond(
            seq,
            "scopes",
            body={"scopes": [scope.model_dump(by_alias=True)]},
        )

    async def _on_variables(self, seq: int, args: dict[str, Any]) -> None:
        ref = int(args.get("variablesReference", 0))
        if ref != self.SCOPE_REFERENCE:
            await self._respond(seq, "variables", body={"variables": []})
            return

        ctx = self._current_ctx
        variables: list[Variable] = []
        if ctx is not None:
            variables.extend(self._snapshot_variables(ctx))

        await self._respond(
            seq,
            "variables",
            body={
                "variables": [v.model_dump(by_alias=True) for v in variables],
            },
        )

    async def _on_evaluate(self, seq: int, args: dict[str, Any]) -> None:
        """Resolve `expression` against the held DebugContext.

        Limited to the same field set as the `variables` view —
        `turn_index`, `message_count`, `last_call.name`,
        `last_call.arguments`, `pending_mutation.role`. Arbitrary-
        expression evaluation is intentionally out of scope; the
        interactive REPL is the surface for that.
        """
        expression = str(args.get("expression", "")).strip()
        ctx = self._current_ctx
        if ctx is None:
            await self._respond(
                seq, "evaluate", success=False, message="no active breakpoint to evaluate against"
            )
            return

        snapshot = {v.name: v for v in self._snapshot_variables(ctx)}
        if expression not in snapshot:
            supported = ", ".join(sorted(snapshot.keys())) or "(none)"
            await self._respond(
                seq,
                "evaluate",
                success=False,
                message=f"unsupported expression {expression!r}; supported names: {supported}",
            )
            return

        var = snapshot[expression]
        await self._respond(
            seq,
            "evaluate",
            body={
                "result": var.value,
                "type": var.type,
                "variablesReference": 0,
            },
        )

    async def _on_source(self, seq: int, args: dict[str, Any]) -> None:
        ref = args.get("sourceReference") or args.get("source", {}).get("sourceReference")
        if ref != self.SOURCE_REFERENCE:
            await self._respond(seq, "source", success=False, message="unknown sourceReference")
            return

        lines = self._synthesized_lines()
        await self._respond(
            seq,
            "source",
            body={
                "content": "\n".join(lines) + ("\n" if lines else ""),
                "mimeType": "text/plain",
            },
        )

    async def _on_terminate(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(seq, "terminate")
        await self._shutdown_session(reason="terminate")

    async def _on_disconnect(self, seq: int, args: dict[str, Any]) -> None:
        await self._respond(seq, "disconnect")
        await self._shutdown_session(reason="disconnect")
        self._terminated = True

    # ------------------------------------------------------------------ breakpoint pump

    async def _on_breakpoint(self, ctx: DebugContext) -> None:
        """Wired into `DebugRunner.breakpoint_callback`. Parks on
        `_continue_event` until the editor decides what to do.
        """
        self._current_ctx = ctx
        self._continue_event.clear()
        await self._send_event(
            "stopped",
            body={
                "reason": "breakpoint",
                "threadId": self.THREAD_ID,
                "allThreadsStopped": True,
            },
        )
        try:
            await self._continue_event.wait()
        finally:
            self._current_ctx = None
            # If the editor disconnected while we were parked, ensure the
            # context is in a defined state so DebugRunner can return
            # cleanly. Default to resume — a hard kill came in via
            # disconnect/terminate which already set abort.
            if not ctx.aborted and not ctx.resumed:
                ctx.resume()

    async def _resume_breakpoint(self) -> None:
        """Common code for continue/next/stepIn/stepOut: emit `continued`,
        then unblock the parked breakpoint so the runner can return.
        """
        if self._current_ctx is None:
            return
        await self._send_event(
            "continued",
            body={"threadId": self.THREAD_ID, "allThreadsContinued": True},
        )
        self._continue_event.set()

    # ------------------------------------------------------------------ session lifecycle

    async def _run_session_with_lifecycle(self) -> None:
        assert self.run_session is not None
        try:
            await self.run_session()
        except Exception as exc:  # noqa: BLE001 - report to editor and emit terminated
            await self._send_output(
                f"session error: {type(exc).__name__}: {exc}",
                category="stderr",
            )
        finally:
            await self._send_event(
                "terminated",
                body={},
            )
            await self._send_event("exited", body={"exitCode": 0})

    async def _shutdown_session(self, *, reason: str) -> None:
        """Terminate or disconnect: abort the current breakpoint, cancel
        the session task, drain it.
        """
        if self._current_ctx is not None:
            self._current_ctx.abort()
            self._continue_event.set()
        await self._cleanup_session()
        if reason == "disconnect":
            await self._send_event("terminated", body={})

    async def _cleanup_session(self) -> None:
        task = self._session_task
        if task is None or task.done():
            return
        task.cancel()
        # Cleanup phase — swallow whatever the cancelled session task
        # surfaces (CancelledError, or a downstream exception that the
        # session was about to propagate). The editor already heard
        # `terminated` if the session finished cleanly; if it didn't,
        # the editor disconnected so there's no one to tell anyway.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # ------------------------------------------------------------------ trajectory source

    def _synthesized_lines(self) -> list[str]:
        if self.synthesize_source is None:
            return []
        return list(self.synthesize_source())

    def _max_line(self) -> int:
        lines = self._synthesized_lines()
        return max(len(lines), 1)

    def _snapshot_variables(self, ctx: DebugContext) -> list[Variable]:
        out: list[Variable] = [
            Variable(name="turn_index", value=str(ctx.turn_index), type="int"),
            Variable(name="message_count", value=str(len(ctx.messages)), type="int"),
        ]
        if ctx.last_call is not None:
            out.append(
                Variable(
                    name="last_call.name",
                    value=ctx.last_call.name,
                    type="str",
                )
            )
            out.append(
                Variable(
                    name="last_call.arguments",
                    value=repr(ctx.last_call.arguments),
                    type="dict",
                )
            )
        if ctx.pending_mutation is not None:
            out.append(
                Variable(
                    name="pending_mutation.role",
                    value=ctx.pending_mutation.role,
                    type="str",
                )
            )
        return out

    # ------------------------------------------------------------------ outbound primitives

    def _next_seq(self) -> int:
        self._out_seq += 1
        return self._out_seq

    async def _respond(
        self,
        request_seq: int,
        command: str,
        *,
        success: bool = True,
        body: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        envelope: dict[str, Any] = {
            "seq": self._next_seq(),
            "type": "response",
            "request_seq": request_seq,
            "success": success,
            "command": command,
        }
        if body is not None:
            envelope["body"] = body
        if message is not None:
            envelope["message"] = message
        await self._write(envelope)

    async def _send_event(self, event: str, body: dict[str, Any] | None = None) -> None:
        envelope: dict[str, Any] = {
            "seq": self._next_seq(),
            "type": "event",
            "event": event,
        }
        if body is not None:
            envelope["body"] = body
        await self._write(envelope)

    async def _send_output(self, output: str, *, category: str = "stdout") -> None:
        await self._send_event(
            "output",
            body={"category": category, "output": output + "\n"},
        )

    async def _write(self, envelope: dict[str, Any]) -> None:
        if self._writer is None:
            return
        await write_message(self._writer, envelope)
