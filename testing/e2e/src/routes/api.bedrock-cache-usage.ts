import { createFileRoute } from '@tanstack/react-router'
import { chat } from '@tanstack/ai'
import { BedrockConverseTextAdapter } from '@tanstack/ai-bedrock'

class CacheUsageAdapter extends BedrockConverseTextAdapter<'us.amazon.nova-pro-v1:0'> {
  protected override async sendStream() {
    return (async function* () {
      yield { messageStart: { role: 'assistant' as const } }
      yield {
        contentBlockDelta: { delta: { text: 'hello' }, contentBlockIndex: 0 },
      }
      yield { messageStop: { stopReason: 'end_turn' as const } }
      yield {
        metadata: {
          usage: {
            inputTokens: 4,
            outputTokens: 165,
            totalTokens: 1945,
            cacheReadInputTokens: 1000,
            cacheWriteInputTokens: 776,
          },
          metrics: { latencyMs: 1 },
        },
      }
    })()
  }
}

export const Route = createFileRoute('/api/bedrock-cache-usage')({
  server: {
    handlers: {
      POST: async () => {
        const adapter = new CacheUsageAdapter(
          { apiKey: 'fixture' },
          'us.amazon.nova-pro-v1:0',
        )
        for await (const chunk of chat({
          adapter,
          messages: [{ role: 'user', content: 'hello' }],
        })) {
          if (chunk.type === 'RUN_FINISHED') return Response.json(chunk.usage)
        }
        return new Response('Missing usage', { status: 500 })
      },
    },
  },
})
