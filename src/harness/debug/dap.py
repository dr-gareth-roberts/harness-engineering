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
    adapter.attach_hooks(hooks)  # M3.6 — enables frame-aware stepping
    adapter.run_session = lambda: _drive(orchestrator, record)
    adapter.synthesize_source = lambda: _trajectory_lines(record)
    await adapter.serve(reader, writer)

DAP subset implemented:

- Requests: initialize, launch, setBreakpoints, configurationDone,
  threads, stackTrace, scopes, variables, evaluate (limited), source,
  continue, next, stepIn, stepOut, pause, terminate, disconnect.
- Events: initialized, stopped, continued, output, terminated, exited.

Stepping and pause (M3.6 / Wave 13b): the harness execution model
distinguishes two frame kinds — `orchestrator` (between tool
dispatches, including the assistant message that triggers them) and
`tool` (inside a single tool dispatch, between `PreToolUse` and
`PostToolUse`). The three step requests have distinct semantics:

- `next` (step_over): run to the next turn boundary, ignoring tool
  dispatches in between. `break_on_predicate` fires at the next
  `DebugRunner.__call__`. This is the pre-1.1.0 behavior that the
  other two step kinds also aliased to; M3.6 splits them apart.
- `stepIn` (step_in): run until the next `PreToolUse` event, then
  pause inside the tool frame. Fallback: if no further tool dispatch
  happens before the next turn boundary, pause at that turn boundary
  instead — so the editor's step-in button is never silently
  unresponsive.
- `stepOut` (step_out): from inside a tool frame, run until the
  current dispatch's `PostToolUse` fires, then pause at the next
  event (another `PreToolUse` in the same tool-use loop, or the next
  turn boundary). From outside a tool frame, step_out has no outer
  frame to return to — it falls back to step_over semantics.

`pause` sets `_pause_requested` so `break_on_predicate` (and the
hook-based pause path) fires unconditionally at the next
opportunity, making the editor's pause button responsive
mid-trajectory.

Frame tracking requires the adapter to observe `PreToolUse` /
`PostToolUse` events directly: `DebugRunner.break_on` runs only at
turn boundaries, so a tool-frame pause point can't come from
`break_on`. Callers wire this with `adapter.attach_hooks(hooks)`,
registering listeners that update `_current_frame` and, when the
step mode demands it, synthesize a breakpoint by invoking
`breakpoint_callback` from within the hook handler. If
`attach_hooks` is never called (legacy wiring), `stepIn` /
`stepOut` degrade to step_over — `break_on_predicate` still fires
at the next turn boundary, so the editor's button isn't ignored;
it just operates at coarser granularity.

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
from typing import TYPE_CHECKING, Any, Literal

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
    from harness.hooks.events import PostToolUse, PreToolUse
    from harness.hooks.runner import HookRunner

BreakpointCallback = Callable[["DebugContext"], Awaitable[None]]
BreakPredicate = Callable[["DebugContext"], bool]
SourceProvider = Callable[[], list[str]]
SessionRunner = Callable[[], Awaitable[None]]

# Step modes the DAP adapter understands. See module docstring for the
# precise semantics of each:
# - `step_over` (DAP `next`): run to next turn boundary.
# - `step_in`  (DAP `stepIn`): run to next `PreToolUse`, then pause.
# - `step_out` (DAP `stepOut`): from tool frame, run past current
#   `PostToolUse`, pause on the next event.
StepMode = Literal["step_over", "step_in", "step_out"]

# The execution frame the adapter believes the session is currently
# in. `None` before the session has fired any tool events; the
# adapter conservatively treats `None` as "orchestrator" for stepping
# decisions (step_out has no inner frame to leave).
Frame = Literal["orchestrator", "tool"]


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

    def __init__(self, *, allow_evaluate: bool = False) -> None:
        # Breakpoint coordination — see module docstring on concurrency.
        self._continue_event = asyncio.Event()
        self._current_ctx: DebugContext | None = None

        # Set of 0-based turn_index values where the editor wants to break.
        self._breakpoint_turns: set[int] = set()

        # Wave 13b #16 — pause-on-demand. When True, the next `break_on`
        # check fires unconditionally so the next runner invocation
        # pauses. Cleared automatically once consumed.
        self._pause_requested = False
        # M3.6 — step semantics. Each step kind drives a different
        # break point; see module docstring for the full table. The
        # field clears once the matching break fires. `None` means no
        # step is in progress (just `continue` or a freshly-set
        # breakpoint).
        self._step_mode: StepMode | None = None
        # M3.6 — execution frame tracking. Updated by the `PreToolUse`
        # / `PostToolUse` hook listeners registered via
        # `attach_hooks`. `None` means we have no signal yet (no tool
        # event has been observed); stepping treats that as
        # "orchestrator" since the orchestrator frame is the natural
        # default when the session begins.
        self._current_frame: Frame | None = None
        # M3.6 — when step_out fires while inside a tool frame, this
        # flag is set in the `PostToolUse` handler so the next event
        # (the next `PreToolUse` of a later call, or the next turn
        # boundary, whichever comes first) becomes a breakpoint. It
        # clears on consumption.
        self._break_on_next_event = False

        # M3.6 — most recent `ToolCall` observed via `PreToolUse`,
        # cached so the hook-synthesized DebugContext at the tool
        # frame has a `last_call` (the same field the orchestrator
        # frame already populates). Reset on `PostToolUse`.
        self._active_tool_call: Any = None  # ToolCall | None — Any keeps imports lazy.

        # Most recent turn index observed at a turn-boundary check.
        # `_break_in_tool_frame` synthesizes a `DebugContext` for the
        # hook-driven pause path and needs a turn_index that reflects
        # the real session progress — otherwise the DAP `stackTrace`
        # response would map every tool-frame pause to source line 1
        # regardless of how deep into the trajectory we are. Updated
        # in two places (belt-and-suspenders):
        #   1. Every consult of `break_on_predicate` — covers the
        #      pause-mid-trajectory case where no turn-boundary
        #      breakpoint has fired yet.
        #   2. Every entry to `_on_breakpoint` — covers the case where
        #      a turn-boundary breakpoint fired and the editor then
        #      stepped into a tool frame.
        self._last_known_turn_index: int = 0

        # Outgoing message sequence. DAP requires a strictly increasing
        # `seq` on every adapter→editor message.
        self._out_seq = 0

        # Transport, set in `serve`.
        self._writer: asyncio.StreamWriter | None = None

        # Session state.
        self._session_task: asyncio.Task[None] | None = None
        self._launched = False
        self._terminated = False

        # M1.8 — guard against duplicate lifecycle events on disconnect.
        # Both the session task's `finally` block and `_shutdown_session`
        # (under `reason="disconnect"`) want to emit `terminated`; whichever
        # fires first sets this, the other no-ops. Same discipline applied
        # to `exited` so the same guard protects both lifecycle endpoints
        # uniformly across the disconnect and terminate paths.
        self._terminated_emitted: bool = False
        self._exited_emitted: bool = False

        # Wave 13b #17 — opt-in arbitrary expression evaluation in DAP
        # `evaluate`. Off by default; the editor passes `allowEvaluate:
        # true` in the launch arguments to enable. When on, the
        # `evaluate` handler routes through the same expression-
        # resolution path the interactive REPL uses.
        self._allow_evaluate_default = allow_evaluate
        self._allow_evaluate = allow_evaluate

        # Caller-supplied wiring.
        self.run_session: SessionRunner | None = None
        self.synthesize_source: SourceProvider | None = None

    # ------------------------------------------------------------------ wiring

    @property
    def break_on_predicate(self) -> BreakPredicate:
        """Closure that consults `_breakpoint_turns`, the pause flag, and
        the step-mode flag. Pass to `DebugRunner`'s `break_on=` so
        `setBreakpoints` / `pause` / `next`-`stepIn`-`stepOut` can
        mutate state without rebuilding the runner.

        `DebugRunner` calls this once per `__call__` (per turn
        boundary). It does not fire mid-tool-dispatch — that's what
        the `attach_hooks` PreToolUse / PostToolUse listeners are
        for.

        Order of precedence:
        - `pause` requested → fire (and clear the flag).
        - `_break_on_next_event` set (step_out aftermath when the
          next event is a turn boundary) → fire.
        - `step_over` set → fire (turn-boundary semantics; the
          natural granularity of `break_on`).
        - `step_in` / `step_out` set → fire as a graceful fallback.
          The hook listeners are responsible for the precise
          frame-aware pause; if no tool dispatch happens between this
          step request and the next turn boundary, the editor's step
          button should still pause *somewhere* rather than silently
          ignoring the click.
        - Otherwise, the per-turn breakpoints from setBreakpoints.
        """

        def _break(ctx: DebugContext) -> bool:
            # Track the most recent real turn_index every consult, so a
            # subsequent hook-driven pause inside a tool frame can
            # synthesize a `DebugContext` whose `turn_index` matches
            # actual session progress. Updated on every call (not just
            # when the predicate returns True) so a pause-button press
            # mid-trajectory — before any breakpoint has fired — still
            # produces an accurate source-line mapping.
            self._last_known_turn_index = ctx.turn_index
            if self._pause_requested:
                # Consume the request once it fires so a follow-up
                # `continue` doesn't immediately re-pause.
                self._pause_requested = False
                return True
            if self._break_on_next_event:
                # step_out aftermath: PostToolUse already fired and we
                # arrived at the next turn boundary without seeing
                # another PreToolUse in between. Pause here as the
                # fallback "next event" target.
                self._break_on_next_event = False
                return True
            if self._step_mode == "step_over":
                # Step over a tool call (next): pause at the next
                # runner invocation. Cleared on consumption.
                self._step_mode = None
                return True
            if self._step_mode in ("step_in", "step_out"):
                # Fallback path — see module docstring. The hook
                # listeners are the primary handler; this catches the
                # "no further tool dispatch" case so a step button is
                # never silently ignored.
                self._step_mode = None
                return True
            return ctx.turn_index in self._breakpoint_turns

        return _break

    @property
    def breakpoint_callback(self) -> BreakpointCallback:
        """The async callback `DebugRunner` should fire when a breakpoint
        hits. Parks until the editor sends `continue` or `disconnect`.
        """
        return self._on_breakpoint

    # ------------------------------------------------------------------ hooks

    def attach_hooks(self, hooks: HookRunner) -> None:
        """Register `PreToolUse` / `PostToolUse` listeners so the adapter
        can track the current execution frame and synthesize tool-frame
        breakpoints for `stepIn` / `stepOut`.

        Wire after constructing the `HookRunner` and before starting
        the orchestrator session. Calling this is required for true
        frame-aware stepping; if omitted, `stepIn` / `stepOut`
        degrade gracefully to step-over (the runner's break_on
        predicate still fires at the next turn boundary, so the
        editor button isn't silently ignored — it just operates at
        coarser granularity).

        Pre-1.1.0 behavior: `stepIn` / `stepOut` had no hook
        listeners at all and were hard-aliased to step_over. M3.6
        fixed this for both wiring paths: the new hook listeners
        track frame state, and `break_on_predicate` carries the
        no-tool-dispatch fallback.
        """
        # Local import to keep top-level imports lean and avoid a
        # circular import on `harness.hooks.events` (which transitively
        # depends on `harness.prompts.messages`).
        from harness.hooks.events import PostToolUse as _PostToolUse
        from harness.hooks.events import PreToolUse as _PreToolUse

        hooks.register(_PreToolUse, self._on_pre_tool_use)
        hooks.register(_PostToolUse, self._on_post_tool_use)

    async def _on_pre_tool_use(self, event: PreToolUse) -> None:
        """`PreToolUse` listener. Enters the tool frame and, if the
        editor asked for a tool-frame pause (step_in, or a step_out
        whose successor is another tool call), synthesizes a
        breakpoint by invoking `breakpoint_callback` directly.

        Returns no `HookDecision` — this listener is purely
        observational from the hook runner's perspective (a paused
        tool dispatch still resumes through the same dispatch path
        once the editor sends `continue`).
        """
        self._current_frame = "tool"
        self._active_tool_call = event.call

        should_break = False
        if self._step_mode == "step_in":
            self._step_mode = None
            should_break = True
        elif self._break_on_next_event:
            # step_out's PostToolUse handler set this — pause at the
            # next PreToolUse if there is one before the next turn
            # boundary.
            self._break_on_next_event = False
            should_break = True
        elif self._pause_requested:
            # Editor pressed pause while mid-tool-loop — honor it at
            # the next PreToolUse rather than waiting for the next
            # turn boundary.
            self._pause_requested = False
            should_break = True

        if should_break:
            await self._break_in_tool_frame(event.call)

    async def _on_post_tool_use(self, event: PostToolUse) -> None:
        """`PostToolUse` listener. Leaves the tool frame. If the editor
        asked for step_out from this frame, arm `_break_on_next_event`
        so the very next event (another PreToolUse or the next turn
        boundary) pauses.
        """
        self._current_frame = "orchestrator"
        self._active_tool_call = None

        if self._step_mode == "step_out":
            # The pause point for step_out is the *next* event after
            # the current dispatch completes. Disarm the step flag and
            # arm the next-event trap.
            self._step_mode = None
            self._break_on_next_event = True

    async def _break_in_tool_frame(self, call: Any) -> None:
        """Synthesize a `DebugContext` pinned to the current tool frame
        and route through the same `_on_breakpoint` parking path the
        turn-boundary breakpoints use.

        The context is intentionally minimal — `last_call` is the
        tool we're entering, `turn_index` carries the most recent
        turn index observed at a turn boundary (or the most recent
        `_on_breakpoint` entry), and `messages` is empty.

        `turn_index` is *not* hard-coded to 0: the DAP `stackTrace`
        response maps `ctx.turn_index + 1` directly to the displayed
        source line, so a hard-coded zero would mislocate every
        tool-frame pause to source line 1 regardless of how deep into
        the trajectory we are. `_last_known_turn_index` is kept in
        sync by `break_on_predicate` (every consult) and
        `_on_breakpoint` (every turn-boundary pause), so by the time
        a `PreToolUse`-driven pause synthesizes this context the
        value reflects actual session progress.

        Editors typically read the other fields through `variables` /
        `evaluate`; the values are honest about what's known at this
        point (we're between turn-boundary checkpoints).
        """
        # Local import to avoid a top-level dependency cycle.
        from harness.debug.context import DebugContext

        ctx = DebugContext([], last_call=call, turn_index=self._last_known_turn_index)
        await self._on_breakpoint(ctx)

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
        # Wave 13b #17 — opt-in arbitrary expression evaluation. The
        # editor passes `allowEvaluate: true` in the launch args to
        # enable. Default falls back to whatever was set at adapter
        # construction time.
        if "allowEvaluate" in args:
            self._allow_evaluate = bool(args["allowEvaluate"])
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
        # M3.6 — step over the next tool call. The `break_on`
        # predicate fires at the next turn boundary regardless of how
        # many tool dispatches occur in between, so the editor's
        # "next" button skips all of them.
        self._step_mode = "step_over"
        await self._respond(seq, "next")
        await self._resume_breakpoint()

    async def _on_stepIn(self, seq: int, args: dict[str, Any]) -> None:
        # M3.6 — step into the next tool dispatch. The `PreToolUse`
        # hook listener (registered via `attach_hooks`) synthesizes a
        # breakpoint inside the tool frame the moment the next tool
        # call begins. Fallback: if no tool dispatch happens before
        # the next turn boundary, `break_on_predicate` pauses there
        # so the button isn't silently ignored.
        self._step_mode = "step_in"
        await self._respond(seq, "stepIn")
        await self._resume_breakpoint()

    async def _on_stepOut(self, seq: int, args: dict[str, Any]) -> None:
        # M3.6 — step out of the current frame. From a tool frame
        # (paused inside a tool dispatch), the `PostToolUse` listener
        # arms `_break_on_next_event` so the very next event — either
        # another tool call's `PreToolUse` or the next turn boundary
        # — pauses. From an orchestrator frame (no outer frame to
        # return to), step_out falls back to step_over: pause at the
        # next turn boundary. See module docstring for the table.
        if self._current_frame == "tool":
            self._step_mode = "step_out"
        else:
            # Graceful fallback — there's no "outer" frame to return
            # to from the orchestrator, so step_out matches step_over
            # in this case.
            self._step_mode = "step_over"
        await self._respond(seq, "stepOut")
        await self._resume_breakpoint()

    async def _on_pause(self, seq: int, args: dict[str, Any]) -> None:
        # Wave 13b #16 — pause on demand. Set the pause flag; the next
        # `break_on` check fires unconditionally and the runner stops
        # at the next opportunity (typically the next iteration of
        # the tool-use loop). The editor's pause button now works.
        # If we're currently mid-breakpoint there's nothing to do —
        # the editor sees the existing `stopped` event.
        self._pause_requested = True
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

        Two modes:

        - **Default** — limited to the same field set the `variables`
          view exposes (`turn_index`, `message_count`,
          `last_call.name`, `last_call.arguments`,
          `pending_mutation.role`). Safe surface; what most editor
          users want.
        - **`allowEvaluate: true`** in the launch arguments (Wave 13b
          #17) — routes through the same code path as the REPL's
          `inspect` command, accepting arbitrary Python expressions
          against `ctx`. Same security trade-off as the REPL: only
          reachable when a breakpoint hits in an opt-in debug session,
          never in production paths. Documented as the explicit
          opt-in for editor users who want REPL-equivalent power.
        """
        expression = str(args.get("expression", "")).strip()
        ctx = self._current_ctx
        if ctx is None:
            await self._respond(
                seq, "evaluate", success=False, message="no active breakpoint to evaluate against"
            )
            return

        if self._allow_evaluate:
            from harness.debug.repl import evaluate_in_context

            try:
                value = evaluate_in_context(expression, ctx)
            except Exception as exc:  # noqa: BLE001 - surface to editor as a failed response
                await self._respond(
                    seq,
                    "evaluate",
                    success=False,
                    message=f"{type(exc).__name__}: {exc}",
                )
                return
            await self._respond(
                seq,
                "evaluate",
                body={
                    "result": repr(value),
                    "variablesReference": 0,
                },
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
        # Keep the last-known turn index in sync so a subsequent
        # `stepIn` into a tool frame synthesizes a `DebugContext` with
        # the correct `turn_index` (and therefore the right DAP source
        # line). Covers the common flow: turn-boundary break → editor
        # presses stepIn → `_break_in_tool_frame` fires.
        self._last_known_turn_index = ctx.turn_index
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
            # Idempotent emit (M1.8) — `_shutdown_session(reason="disconnect")`
            # may have already raced ahead and emitted these, or may be
            # about to. The flags ensure the editor sees `terminated` +
            # `exited` exactly once per session lifecycle.
            await self._emit_terminated_once()
            await self._emit_exited_once()

    async def _shutdown_session(self, *, reason: str) -> None:
        """Terminate or disconnect: abort the current breakpoint, cancel
        the session task, drain it.

        Lifecycle events are routed through `_emit_*_once` helpers so
        whichever of this method or the session task's `finally` runs
        first wins, and the other no-ops. Same discipline on both the
        disconnect and terminate paths.
        """
        if self._current_ctx is not None:
            self._current_ctx.abort()
            self._continue_event.set()
        await self._cleanup_session()
        if reason == "disconnect":
            await self._emit_terminated_once()

    async def _emit_terminated_once(self) -> None:
        """Emit the `terminated` lifecycle event at most once per session.

        DAP requires `terminated` so the editor can wind down its UI;
        emitting it twice (which the legacy disconnect path did — once
        from the session task's `finally`, once from `_shutdown_session`)
        confuses editors that track state machines. See M1.8.
        """
        if self._terminated_emitted:
            return
        self._terminated_emitted = True
        await self._send_event("terminated", body={})

    async def _emit_exited_once(self) -> None:
        """Emit the `exited` lifecycle event at most once per session.

        Symmetric to `_emit_terminated_once`; same uniform guard so any
        future call site can rely on at-most-once semantics without
        threading the flag through call paths.
        """
        if self._exited_emitted:
            return
        self._exited_emitted = True
        await self._send_event("exited", body={"exitCode": 0})

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
