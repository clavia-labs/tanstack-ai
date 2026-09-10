---
'@tanstack/ai-anthropic': patch
'@tanstack/ai-bedrock': patch
---

Include cache reads and writes in prompt token counts. Include them in Anthropic total tokens and preserve the already-inclusive Bedrock total. Keep cache details available for pricing.
