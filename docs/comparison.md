# How does this compare to LangChain / DSPy / AutoGen / CrewAI?

Honest placement on the spectrum. None of these libraries are wrong;
they target different audiences with different trade-offs. This page
tells you when **harness-engineering** is the right pick — and when
something else fits better.

## TL;DR

| You want | Pick |
|---|---|
| The fastest path from "I have an API key" to a chatbot | **LangChain** |
| Optimization-driven prompt programs (DSPy compiler, signatures, evaluations) | **DSPy** |
| Orchestrating *multiple* agents talking to each other | **AutoGen** or **CrewAI** |
| A typed, reproducible **harness** around a single agent — tool dispatch, hooks, replay, observability, debug, privacy, contracts — without buying into a framework | **harness-engineering** |
| To rip parts out and use them with someone else's agent | **harness-engineering** |

## What "harness" means in the name

A model alone produces tokens; the **harness** is what turns those
tokens into an agent: tools the model can call, prompts it sees,
policies that constrain its tool use, memory that persists between
turns, replay so you can debug, telemetry so you know what happened,
boundaries so secrets don't leak.

`harness-engineering` is the library of those primitives. It has no
opinion on prompt construction, on orchestration patterns
(supervisor/worker, chain-of-thought, etc.), or on multi-agent
choreography. It gives you small protocols and lets you wire them
together.

## Concrete differences

### vs LangChain (and LangGraph / LangSmith)

LangChain optimizes for *breadth*: every model provider, every
vector DB, every tool, every chain pattern, all in one ecosystem.
The modern split — **LangGraph** for typed state-machine
orchestration, **LangSmith** for tracing and evaluation — addresses
the historical complaints about loose types and opaque execution;
that ecosystem is now closer to "framework you build inside" than
"library you import primitives from."

`harness-engineering` sits a level lower. Small protocols (`Runner`,
`Sink`, `MemoryStore`, `Detector`, `Predictor`), strict types
(`mypy --strict` passes across `src/` and `tests/`), and
vendor-specific code lives in vendor-specific files behind optional
extras. There's no DSL — just
`Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]`,
and `AnthropicRunner` / `OpenAICompatRunner` implement it.

**Pick LangChain / LangGraph / LangSmith** if you want a turnkey
ecosystem and you're comfortable adopting the framework's idioms,
graph model, and tracing UI. **Pick harness-engineering** if you'd
rather assemble the runtime from small protocols you can swap or
reuse without owning the whole stack.

### vs DSPy

DSPy is a different beast: it treats prompts as a program you
*compile*, with signatures, modules, optimizers, and an evaluation
loop that fine-tunes prompt construction itself. You write what
you want done; DSPy figures out the prompt.

`harness-engineering` doesn't optimize prompts. It assumes you
write the prompt and the tool definitions; its job is to give you
a controllable, observable runtime around that.

The two are complementary: a DSPy program *outputs* an agent
specification (system prompt + tools + a chain of calls).
`harness-engineering` is one of the runtimes that specification
could be executed on. There's no integration today, but the surfaces
don't fight.

**Pick DSPy** if your problem is "the prompt itself isn't optimal
and I want to optimize it." **Pick harness-engineering** if your
problem is "I have a prompt, and now I need a robust runtime
around it."

### vs AutoGen

AutoGen's core abstraction is the *conversation between agents*:
agents (and, in modern AutoGen, teams of agents) exchange messages
to solve a task, with tool dispatch, memory, and orchestration
patterns built on top of that primitive.

`harness-engineering` doesn't have an opinion on multi-agent
patterns. Its `Orchestrator` runs *one* sub-agent through a
tool-use loop. If you want supervisor/worker, voting, recursive
sub-agents — build it on top, or use AutoGen.

**Pick AutoGen** when the problem is naturally multi-agent (e.g.,
code review with reviewer + executor + critic). **Pick
harness-engineering** when you want a single-agent runtime with
serious observability, and you'll handle multi-agent choreography
yourself if you need it.

### vs CrewAI

CrewAI is similar to AutoGen but with stronger opinions about
roles and processes (a "Crew" of agents, each with a defined role,
working through a sequential or hierarchical process). The
abstraction is higher-level than AutoGen's; less code to wire up
the common case, less control over the rare case.

`harness-engineering` doesn't compete here at all — it's at a lower
level. You'd build a CrewAI-equivalent on top of it if you wanted
roles + process; the underlying primitives (Runner, Hook, Memory)
would carry over.

**Pick CrewAI** when you want a high-level "crew of role-based
agents" abstraction and you don't want to design the orchestration
yourself. **Pick harness-engineering** when you want primitives.

## What `harness-engineering` is *bad* at

Honest list, not bullet-armor:

- **Greenfield chatbot demos** — there's no `from harness import
  ChatBot; bot = ChatBot.from_openai("gpt-4o")`. You write the
  tools, the prompts, the dispatcher. LangChain gives you something
  faster.
- **Multi-agent patterns** — no `GroupChat`, no `RoleHierarchy`. If
  your design needs more than one agent talking to another, you're
  building that yourself on top.
- **Prompt optimization** — DSPy is the answer if "the prompt isn't
  good enough" is your problem.
- **No-code / low-code** — every example is Python, every protocol
  is typed, every tool is a Pydantic model. Operators who don't
  write code aren't the audience.
- **Vector DB integration** — the `MemoryStore` Protocol is
  ToolResult / SessionRecord shaped, not "documents and embeddings."
  For RAG you wire up the retrieval yourself and pass results into
  a tool. Not bad, just not built-in.

## What `harness-engineering` is *good* at

- **Reproducibility.** `ReplayRunner` over recorded
  `SessionRecord`s, deterministic fakes (`EchoRunner`,
  `CannedRunner`), counterfactual replays, and a cross-provider
  `diff_eval` matrix. Vendor HTTP-level cassettes are not in scope
  yet (listed as deferred on the [Roadmap](roadmap.md)). 565 tests
  run in under 5 seconds; the entire CI matrix gates strictly.
- **Observability you can rely on.** `Sink` Protocol with
  `JSONLSink`, `MemorySink`, `OpenTelemetrySink` (under `[otel]`).
  Trace_id / span_id correlation propagated automatically through
  the orchestrator and dispatcher. Your existing OTel backend
  works without custom code.
- **Type safety.** `mypy --strict` passes across `src/` and `tests/`
  (163 source files at 1.0). Tools are Pydantic, hooks are typed,
  protocols are explicit. Refactors don't silently break the world.
- **Pluggable everything.** Every seam is a `Protocol` — `Runner`,
  `Sink`, `MemoryStore`, `Detector`, `Predictor`. Drop in your own;
  nothing inherits from a base class.
- **Replaceability.** Use a single module without buying into the
  rest. `from harness.privacy import PrivacyBoundary` works without
  any of the runners or orchestrator. You can wrap LangChain's
  agent in a `PrivacyBoundary` if you want.
- **Honest deferrals.** Things that don't ship are listed as
  deferred with the reason ([Roadmap](roadmap.md)). No silent
  promises; the README and docs aim to match the code.

## "Should I use this in production?"

Yes, if:

- You want strict types + replay + structured telemetry,
- You're already comfortable with Python's async ecosystem,
- You have one agent (not a multi-agent system) that needs solid
  observability and operational tooling,
- You want to keep the model layer swappable (Anthropic today,
  Ollama tomorrow, Bedrock the day after).

No, if:

- You need the surface area LangChain provides,
- You want a multi-agent orchestration framework,
- You don't write Python.

## Where to read next

- [**Quickstart**](quickstart.md) — 10 minutes to a working agent.
- [**Cookbook**](cookbook/index.md) — concrete recipes for the
  features that distinguish this library.
- [**Architecture**](architecture.md) — the protocol seams.
