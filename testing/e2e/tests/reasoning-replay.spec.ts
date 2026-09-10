import { test, expect } from '@playwright/test'
import { StreamProcessor } from '@tanstack/ai'
import type { ModelMessage } from '@tanstack/ai'
import { resolveDebugOption } from '@tanstack/ai/adapter-internals'
import { createAnthropicChat } from '@tanstack/ai-anthropic'
import { createOpenaiChat } from '@tanstack/ai-openai'

for (const provider of ['anthropic', 'openai'] as const) {
  test(`${provider} replays opaque reasoning after history serialization`, async () => {
    const thinking = {
      type: 'thinking',
      thinking: 'Check the source',
      signature: 'signed-state',
    }
    const redacted = {
      type: 'redacted_thinking',
      data: 'opaque-redacted-state',
    }
    const tool = { type: 'tool_use', id: 'call_1', name: 'read', input: {} }
    const reasoning = [
      {
        type: 'reasoning',
        id: 'r1',
        encrypted_content: 'encrypted-1',
        summary: [{ type: 'summary_text', text: 'Check the source' }],
      },
      {
        type: 'reasoning',
        id: 'r2',
        encrypted_content: 'encrypted-2',
        summary: [],
      },
    ]
    const call = {
      type: 'function_call',
      id: 'fc_1',
      call_id: 'call_1',
      name: 'read',
      arguments: '{}',
    }
    const events: Array<Record<string, unknown>> =
      provider === 'anthropic'
        ? [
            {
              type: 'content_block_start',
              index: 0,
              content_block: { type: 'thinking', thinking: '', signature: '' },
            },
            {
              type: 'content_block_delta',
              index: 0,
              delta: { type: 'thinking_delta', thinking: thinking.thinking },
            },
            {
              type: 'content_block_delta',
              index: 0,
              delta: { type: 'signature_delta', signature: thinking.signature },
            },
            { type: 'content_block_stop', index: 0 },
            { type: 'content_block_start', index: 1, content_block: redacted },
            { type: 'content_block_stop', index: 1 },
            { type: 'content_block_start', index: 2, content_block: tool },
            { type: 'content_block_stop', index: 2 },
            {
              type: 'message_delta',
              delta: { stop_reason: 'tool_use' },
              usage: { output_tokens: 10 },
            },
            { type: 'message_stop' },
          ]
        : [
            ...reasoning.flatMap((item, index) => [
              {
                type: 'response.output_item.added',
                output_index: index,
                item: { type: 'reasoning', id: item.id },
              },
              ...item.summary.map((part) => ({
                type: 'response.reasoning_summary_text.delta',
                item_id: item.id,
                output_index: index,
                summary_index: 0,
                delta: part.text,
              })),
              { type: 'response.output_item.done', output_index: index, item },
            ]),
            { type: 'response.output_item.added', output_index: 2, item: call },
            {
              type: 'response.function_call_arguments.done',
              output_index: 2,
              item_id: call.id,
              arguments: '{}',
            },
            {
              type: 'response.completed',
              response: {
                id: 'response',
                model: 'gpt-5',
                status: 'completed',
                output: [...reasoning, call],
              },
            },
          ]
    const requests: Array<{
      messages?: Array<{ role: string; content: unknown }>
      input?: Array<{ type: string }>
    }> = []
    const fetch: typeof globalThis.fetch = async (input, init) => {
      const request =
        input instanceof Request ? input : new Request(input, init)
      requests.push(
        JSON.parse(
          typeof init?.body === 'string' ? init.body : await request.text(),
        ),
      )
      return new Response(
        events
          .map(
            (event) =>
              `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`,
          )
          .join(''),
        { headers: { 'content-type': 'text/event-stream' } },
      )
    }
    const adapter =
      provider === 'anthropic'
        ? createAnthropicChat('claude-opus-4-1', 'test-key', { fetch })
        : createOpenaiChat('gpt-5', 'test-key', { fetch })
    const logger = resolveDebugOption(false)
    const processor = new StreamProcessor()
    await processor.process(
      adapter.chatStream({
        model: adapter.model,
        messages: [{ role: 'user', content: 'Read the source' }],
        logger,
      }),
    )
    const history: Array<ModelMessage> = JSON.parse(
      JSON.stringify(processor.toModelMessages()),
    )
    history.push({
      role: 'tool',
      toolCallId: 'call_1',
      content: 'Source contents',
    })
    for await (const _chunk of adapter.chatStream({
      model: adapter.model,
      messages: history,
      logger,
    })) {
    }
    expect(requests).toHaveLength(2)
    if (provider === 'anthropic')
      expect(
        requests[1]?.messages?.find((message) => message.role === 'assistant')
          ?.content,
      ).toEqual([thinking, redacted, tool])
    else
      expect(
        requests[1]?.input?.filter((item) => item.type === 'reasoning'),
      ).toEqual(reasoning)
  })
}
