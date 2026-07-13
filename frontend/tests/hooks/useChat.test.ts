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

const askQuestionStreamMock = vi.mocked(api.askQuestionStream);

const ZERO_TIMINGS: TimingsResponse = {
  embed_ms: 0,
  search_ms: 0,
  rerank_ms: 0,
  generate_ms: 0,
  total_ms: 0,
};

afterEach(() => {
  askQuestionStreamMock.mockReset();
});

describe('useChat', () => {
  it('streams sources then deltas into a single assistant turn, then finalizes on done', async () => {
    let resolveStream!: () => void;
    askQuestionStreamMock.mockImplementation((_q, _p, _o, _f, handlers: QuestionStreamHandlers) => {
      handlers.onSources?.([], 'trace-1');
      handlers.onDelta?.('Run ');
      handlers.onDelta?.('docker compose up.');
      return new Promise<void>((resolve) => {
        resolveStream = () => {
          handlers.onDone?.('Run docker compose up.', [], ZERO_TIMINGS, 'trace-1');
          resolve();
        };
      });
    });

    const { result } = renderHook(() => useChat());

    let askPromise!: Promise<void>;
    act(() => {
      askPromise = result.current.ask('How do I run it?');
    });

    expect(result.current.pending).toBe(true);
    expect(result.current.turns).toHaveLength(2);
    expect(result.current.turns[0]).toMatchObject({ role: 'user', content: 'How do I run it?' });
    expect(result.current.turns[1]).toMatchObject({
      role: 'assistant',
      content: 'Run docker compose up.',
      traceId: 'trace-1',
    });

    await act(async () => {
      resolveStream();
      await askPromise;
    });

    expect(result.current.pending).toBe(false);
    expect(result.current.turns).toHaveLength(2);
    expect(result.current.turns[1]).toMatchObject({
      role: 'assistant',
      content: 'Run docker compose up.',
      traceId: 'trace-1',
      timings: ZERO_TIMINGS,
    });
  });

  it('keeps the traceId set by the sources event even if the stream then errors', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, handlers: QuestionStreamHandlers) => {
        handlers.onSources?.([], 'trace-2');
        throw new ApiClientError('Generation provider request failed.');
      },
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('alpha');
    });

    // The failure still appends a separate error turn (existing behavior); the
    // assistant turn created by the sources event keeps its trace link so the
    // partial trace recorded server-side stays reachable from the UI.
    expect(result.current.turns).toHaveLength(3);
    expect(result.current.turns[1]).toMatchObject({ role: 'assistant', traceId: 'trace-2' });
    expect(result.current.turns[2]).toMatchObject({ role: 'error' });
  });

  it('appends an error turn with the ApiClientError message on failure, with no stray assistant turn', async () => {
    askQuestionStreamMock.mockRejectedValue(
      new ApiClientError('Re-ingest documents before querying.', 409),
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('alpha');
    });

    expect(result.current.pending).toBe(false);
    expect(result.current.turns).toHaveLength(2);
    expect(result.current.turns[1]).toMatchObject({
      role: 'error',
      content: 'Re-ingest documents before querying.',
    });
  });

  it('appends a generic error turn for a non-ApiClientError failure', async () => {
    askQuestionStreamMock.mockRejectedValue(new Error('boom'));

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('alpha');
    });

    expect(result.current.turns[1]).toMatchObject({
      role: 'error',
      content: 'Something went wrong.',
    });
  });

  it('ignores a second ask() while one is still in flight (reentrancy guard)', async () => {
    let resolveStream!: () => void;
    askQuestionStreamMock.mockImplementation(() => {
      return new Promise<void>((resolve) => {
        resolveStream = resolve;
      });
    });

    const { result } = renderHook(() => useChat());

    let firstAskPromise!: Promise<void>;
    act(() => {
      firstAskPromise = result.current.ask('first');
      void result.current.ask('second');
    });

    expect(askQuestionStreamMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveStream();
      await firstAskPromise;
    });

    expect(result.current.pending).toBe(false);
    // Only the first question's user turn exists (no sources/delta/done ever fired
    // for it, so no assistant turn either) — the second call was a no-op.
    expect(result.current.turns).toHaveLength(1);
  });

  it('allows a new ask() once the previous one has resolved', async () => {
    askQuestionStreamMock.mockImplementationOnce(async (_q, _p, _o, _f, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('first answer', [], ZERO_TIMINGS, 'trace-a');
    });
    askQuestionStreamMock.mockImplementationOnce(async (_q, _p, _o, _f, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('second answer', [], ZERO_TIMINGS, 'trace-b');
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('first');
    });
    await act(async () => {
      await result.current.ask('second');
    });

    expect(askQuestionStreamMock).toHaveBeenCalledTimes(2);
    expect(result.current.turns).toHaveLength(4);
    expect(result.current.turns[1]).toMatchObject({ content: 'first answer' });
    expect(result.current.turns[3]).toMatchObject({ content: 'second answer' });
  });

  it('aborts the in-flight request when the component unmounts', async () => {
    askQuestionStreamMock.mockImplementation(
      (_question, _provider, _overrides, _filenames, _handlers, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    );

    const { result, unmount } = renderHook(() => useChat());

    act(() => {
      void result.current.ask('alpha');
    });

    const signal = askQuestionStreamMock.mock.calls[0][5];
    expect(signal?.aborted).toBe(false);

    unmount();

    expect(signal?.aborted).toBe(true);
  });

  it('cancel() aborts the in-flight request and clears pending via the existing error path', async () => {
    askQuestionStreamMock.mockImplementation(
      (_question, _provider, _overrides, _filenames, _handlers, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () =>
            reject(new ApiClientError('Request timed out or was cancelled.')),
          );
        }),
    );

    const { result } = renderHook(() => useChat());

    let askPromise!: Promise<void>;
    act(() => {
      askPromise = result.current.ask('alpha');
    });
    expect(result.current.pending).toBe(true);

    await act(async () => {
      result.current.cancel();
      await askPromise;
    });

    expect(result.current.pending).toBe(false);
    expect(result.current.turns[1]).toMatchObject({
      role: 'error',
      content: 'Request timed out or was cancelled.',
    });
  });

  it('cancel() is a no-op when nothing is in flight', () => {
    const { result } = renderHook(() => useChat());

    expect(() => result.current.cancel()).not.toThrow();
  });
});
