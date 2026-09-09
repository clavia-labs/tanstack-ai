import { describe, expect, it } from 'vitest'
import { buildConverseUsage } from '../../src/converse/usage'

describe('inclusive Converse usage', () => {
  it.each([
    { read: 0, write: 1776 },
    { read: 1776, write: 0 },
    { read: 1000, write: 776 },
  ])(
    'preserves the inclusive total with reads $read and writes $write',
    ({ read, write }) => {
      expect(
        buildConverseUsage({
          inputTokens: 4,
          outputTokens: 165,
          totalTokens: 1945,
          cacheReadInputTokens: read,
          cacheWriteInputTokens: write,
        }),
      ).toEqual({
        promptTokens: 1780,
        completionTokens: 165,
        totalTokens: 1945,
        promptTokensDetails: { cachedTokens: read, cacheWriteTokens: write },
      })
    },
  )

  it('preserves usage without cache fields', () => {
    expect(
      buildConverseUsage({
        inputTokens: 4,
        outputTokens: 165,
        totalTokens: 169,
      }),
    ).toEqual({
      promptTokens: 4,
      completionTokens: 165,
      totalTokens: 169,
    })
  })
})
