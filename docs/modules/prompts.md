# `harness.prompts`

Provider-neutral message + content-block primitives. The
`Message` / `ContentBlock` model normalizes across vendor SDKs;
helpers like `text(...)`, `attach_file(...)`, `attach_image(...)`,
`compact(...)`, and `summarize_compact(...)` cover the common
construction patterns.

## When to reach for this

- You want to construct messages once and run them through different
  runners (Anthropic, OpenAI, replay) without rewriting per vendor.
- You're attaching files (text or Anthropic Files API) or images
  (base64 or URL) to a turn.
- The conversation grew long and you need to drop or summarize old
  turns to fit the context window.

## Quick example

```python
from harness import Message, attach_file, attach_image, compact, text

messages = [
    text("system", "You are concise."),
    text("user", "Summarize the attached file."),
]
messages[1].content.append(attach_file(path="notes.txt"))            # inlines text
messages[1].content.append(attach_image(path="screenshot.png"))      # base64-encoded image

# Trim to last 4 non-system turns when the context grows.
compacted = compact(messages, keep_last=4)
```

## Gotchas

- **`compact()` keeps system messages by default.** Pass
  `include_system=False` to drop them too.
- **`attach_file(file_id="file_...")`** uses Anthropic's Files API
  and only works with `AnthropicRunner`. `OpenAICompatRunner` falls
  back to a `<file file_id=...>` text placeholder.
- **`attach_image(url=...)` requires `media_type=`** — the helper
  can't sniff a remote URL without fetching, so pass it explicitly.
  Path-mode auto-detects from the file extension.
- **`ContentBlock.cache=True`** maps to Anthropic's `cache_control`
  marker. Caps at 4 per request (the runner enforces it client-side
  via `CacheBreakpointLimitExceeded`); has no effect under OpenAI.

## Related

- [`examples/end_to_end.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/end_to_end.py)
- [Cookbook: Cache + speculate](../cookbook/cache-and-speculate.md)
- [`harness.runner`](runner.md) — the consumer of these messages.

## API reference

::: harness.prompts
