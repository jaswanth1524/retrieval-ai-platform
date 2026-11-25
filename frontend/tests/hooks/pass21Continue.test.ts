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

describe('pass 21: Continue', () => {
  it('keeps a save and a note made while it ran when it fails', async () => {
    const { result } = await cutOffAnswer();
    const cut = result.current.turns[1];
    let release: () => void = () => undefined;
    const waiting = new Promise<void>((resolve) => {
      release = resolve;
    });
    stream.mockImplementationOnce(async () => {
      await waiting;
      throw new ApiClientError('The model timed out.', 504);
    });

    let asking: Promise<void> = Promise.resolve();
    act(() => {
      asking = result.current.ask('What comes first?', undefined, undefined, undefined, {
        continueFrom: 'trace-cut',
        replacesAnswer: cut.id,
      });
    });
    act(() => {
      result.current.setTurnSaved(cut.id, true);
    });
    act(() => {
      result.current.setTurnNote(result.current.activeConversationId ?? '', cut.id, 'check with HR');
    });
    await act(async () => {
      release();
      await asking;
    });

    const answer = result.current.turns[1];
    expect(answer.saved).toBeTruthy();
    expect(answer.note).toBe('check with HR');
    expect(answer.continueFailed).toBe('The model timed out.');
    expect(answer.content).toBe('Alpha is first and');
    // The background notice hears it failed, though the last turn is a good answer.
    expect(result.current.lastOutcome).toBe('failed');
  });

  it('keeps the first part’s usage when the continuation reports none', async () => {
    const { result } = await cutOffAnswer();
    const cut = result.current.turns[1];
    stream.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Alpha is first and beta second.', [], TIMINGS, 'trace-2', false, {
        answerStatus: 'answered',
      });
    });
    await act(async () => {
      await result.current.ask('What comes first?', undefined, undefined, undefined, {
        continueFrom: 'trace-cut',
        replacesAnswer: cut.id,
      });
    });
    expect(result.current.turns[1].usage).toEqual({ prompt: 100, completion: 10, costUsd: 0.001 });
    expect(result.current.lastOutcome).toBe('answered');
  });
});
