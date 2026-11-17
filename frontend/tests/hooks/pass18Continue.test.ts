import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useChat } from '../../src/hooks/useChat';
import { ApiClientError, api } from '../../src/api/client';
import type { QuestionStreamHandlers } from '../../src/api/client';
import type { TimingsResponse } from '../../src/api/types';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return { ...actual, api: { ...actual.api, askQuestionStream: vi.fn() } };
});

const stream = vi.mocked(api.askQuestionStream);
const TIMINGS: TimingsResponse = {
  embed_ms: 0,
  search_ms: 0,
  rerank_ms: 0,
  generate_ms: 0,
  total_ms: 0,
  condense_ms: 0,
};

afterEach(() => {
  stream.mockReset();
  localStorage.clear();
});

async function cutOffAnswer() {
  stream.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
    handlers.onDone?.('Alpha is first and', [], TIMINGS, 'trace-cut', false, {
      answerStatus: 'answered',
      truncated: true,
      usage: { prompt_tokens: 100, completion_tokens: 10, cost_usd: 0.001 },
    });
  });
  const hook = renderHook(() => useChat());
  await act(async () => {
    await hook.result.current.ask('What comes first?');
  });
  return hook;
}

describe('continuing a cut-off answer', () => {
  it('writes the rest into the same answer: no second question, usage added up', async () => {
    const { result } = await cutOffAnswer();
    const cut = result.current.turns[1];
    stream.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Alpha is first and beta second [1].', [], TIMINGS, 'trace-2', false, {
        answerStatus: 'answered',
        usage: { prompt_tokens: 120, completion_tokens: 8, cost_usd: 0.002 },
      });
    });

    await act(async () => {
      await result.current.ask('What comes first?', undefined, undefined, undefined, {
        continueFrom: 'trace-cut',
        replacesAnswer: cut.id,
      });
    });

    const turns = result.current.turns;
    expect(turns.map((turn) => turn.role)).toEqual(['user', 'assistant']);
    expect(turns[1].id).toBe(cut.id);
    expect(turns[1].content).toBe('Alpha is first and beta second [1].');
    expect(turns[1].truncated).toBeUndefined();
    expect(turns[1].versions).toBeUndefined();
    expect(turns[1].traceId).toBe('trace-2');
    expect(turns[1].usage).toEqual({ prompt: 220, completion: 18, costUsd: 0.003 });
  });

  it('leaves the answer as it was when its trace is gone, and stops offering Continue', async () => {
    const { result } = await cutOffAnswer();
    const cut = result.current.turns[1];
    stream.mockRejectedValueOnce(new ApiClientError("No trace found with id 'trace-cut'.", 404));

    await act(async () => {
      await result.current.ask('What comes first?', undefined, undefined, undefined, {
        continueFrom: 'trace-cut',
        replacesAnswer: cut.id,
      });
    });

    const turns = result.current.turns;
    expect(turns.map((turn) => turn.role)).toEqual(['user', 'assistant']);
    expect(turns[1].content).toBe('Alpha is first and');
    expect(turns[1].truncated).toBe(true);
    expect(turns[1].cannotContinue).toBe(true);
  });

  it('keeps the answer and says why after another failure', async () => {
    const { result } = await cutOffAnswer();
    const cut = result.current.turns[1];
    stream.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDelta?.(' beta');
      throw new ApiClientError('The model timed out.', 504);
    });

    await act(async () => {
      await result.current.ask('What comes first?', undefined, undefined, undefined, {
        continueFrom: 'trace-cut',
        replacesAnswer: cut.id,
      });
    });

    const answer = result.current.turns[1];
    expect(result.current.turns).toHaveLength(2);
    expect(answer.content).toBe('Alpha is first and');
    expect(answer.cannotContinue).toBeUndefined();
    expect(answer.continueFailed).toBe('The model timed out.');
  });
});
