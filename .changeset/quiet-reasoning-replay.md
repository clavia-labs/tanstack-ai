---
'@tanstack/ai': patch
'@tanstack/ai-anthropic': patch
'@tanstack/openai-base': patch
---

Preserve each reasoning block and its opaque state through tool calls and serialized history. Retain Anthropic redacted blocks and OpenAI encrypted-only items, and attach late signatures to their original thinking steps.
