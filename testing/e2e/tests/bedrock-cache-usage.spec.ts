import { test, expect } from './fixtures'

test('Bedrock Converse includes cache tokens once through chat', async ({
  request,
}) => {
  const response = await request.post('/api/bedrock-cache-usage')
  expect(response.ok()).toBe(true)
  expect(await response.json()).toEqual({
    promptTokens: 1780,
    completionTokens: 165,
    totalTokens: 1945,
    promptTokensDetails: { cachedTokens: 1000, cacheWriteTokens: 776 },
  })
})
