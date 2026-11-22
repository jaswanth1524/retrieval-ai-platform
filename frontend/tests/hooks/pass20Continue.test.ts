import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useChat, withoutRegenerateAttempt } from '../../src/hooks/useChat';
import { api } from '../../src/api/client';
import type { QuestionStreamHandlers } from '../../src/api/client';
import type { TimingsResponse } from '../../src/api/types';
import type { ChatTurn } from '../../src/components/ChatMessage';

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
    handlers.onDone?.('Alpha is first, but', [], TIMINGS, 'trace-cut', false, {
      answerStatus: 'uncited',
      truncated: true,
      unsupportedNumbers: 2,
    });
  });
  const hook = renderHook(() => useChat());
  await act(async () => {
    await hook.result.current.ask('What comes first?');
  });
  return hook;
}

describe('pass 20: continuing an answer', () => {
  it('shows no raw continuation text, keeps the trace id until done, then takes the joined answer', async () => {
    const { result } = await cutOffAnswer();
    const cut = result.current.turns[1];
    const seen: { content: string; traceId: string | null | undefined; continuing: string | null }[] = [];
    let release: () => void = () => undefined;
    stream.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onSources?.([], 'trace-new');
      handlers.onDelta?.('business');
      handlers.onDelta?.(' class [1].');
      await new Promise<void>((resolve) => {
        release = resolve;
      });
      handlers.onDone?.('Alpha is first, but business class [1].', [], TIMINGS, 'trace-new', false, {
        answerStatus: 'answered',
      });
    });

    let asking: Promise<void> = Promise.resolve();
    act(() => {
      asking = result.current.ask('What comes first?', undefined, undefined, undefined, {
        continueFrom: 'trace-cut',
        replacesAnswer: cut.id,
      });
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 120));
    });
    const mid = result.current.turns[1];
    seen.push({ content: mid.content, traceId: mid.traceId, continuing: result.current.continuingTurnId });
    await act(async () => {
      release();
      await asking;
    });

    expect(seen[0]).toEqual({ content: 'Alpha is first, but', traceId: 'trace-cut', continuing: cut.id });
    expect(localStorage.getItem('docrag-chat-history') ?? '').not.toContain('butbusiness');
    const done = result.current.turns[1];
    expect(done.content).toBe('Alpha is first, but business class [1].');
    expect(done.traceId).toBe('trace-new');
    // What described the cut-off fragment no longer applies to the whole answer.
    expect(done.unsupportedNumbers).toBeUndefined();
    expect(done.uncited).toBeUndefined();
    expect(result.current.continuingTurnId).toBeNull();
  });
});

describe('pass 20: a stopped Regenerate', () => {
  it('leaves the old answer and no second copy of its question', () => {
    const turns = [
      { id: 'q1', role: 'user', content: 'Why?' },
      { id: 'a1', role: 'assistant', content: 'Because.' },
      { id: 'q2', role: 'user', content: 'Why?' },
      { id: 'a2', role: 'assistant', content: 'Partly' },
    ] as ChatTurn[];
    expect(withoutRegenerateAttempt(turns, 'a2', 'Why?').map((t) => t.id)).toEqual(['q1', 'a1']);
    // Stopped before its answer existed.
    expect(withoutRegenerateAttempt(turns.slice(0, 3), 'a2', 'Why?').map((t) => t.id)).toEqual([
      'q1',
      'a1',
    ]);
  });
});

describe('pass 20: a failed Regenerate', () => {
  it('keeps the old answer, with why, and no second copy of its question', async () => {
    const { ApiClientError } = await import('../../src/api/client');
    stream.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Because [1].', [], TIMINGS, 'trace-1', false, { answerStatus: 'answered' });
    });
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('Why?');
    });
    const old = result.current.turns[1];
    stream.mockRejectedValueOnce(new ApiClientError('The model timed out.', 504));
    await act(async () => {
      await result.current.ask('Why?', undefined, undefined, undefined, { replacesAnswer: old.id });
    });
    const turns = result.current.turns;
    expect(turns.map((t) => t.role)).toEqual(['user', 'assistant']);
    expect(turns[1]).toMatchObject({ id: old.id, content: 'Because [1].', regenerateFailed: 'The model timed out.' });
  });
});
