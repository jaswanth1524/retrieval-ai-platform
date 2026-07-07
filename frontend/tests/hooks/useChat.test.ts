import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useChat } from '../../src/hooks/useChat';
import { ApiClientError, api } from '../../src/api/client';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return { ...actual, api: { ...actual.api, askQuestion: vi.fn() } };
});

const askQuestionMock = vi.mocked(api.askQuestion);

afterEach(() => {
  askQuestionMock.mockReset();
});

describe('useChat', () => {
  it('appends a user turn, then an assistant turn, toggling pending around the call', async () => {
    let resolveAsk!: (value: { answer: string; sources: [] }) => void;
    askQuestionMock.mockReturnValue(
      new Promise((resolve) => {
        resolveAsk = resolve;
      }),
    );

    const { result } = renderHook(() => useChat());

    let askPromise!: Promise<void>;
    act(() => {
      askPromise = result.current.ask('How do I run it?');
    });

    expect(result.current.pending).toBe(true);
    expect(result.current.turns).toHaveLength(1);
    expect(result.current.turns[0]).toMatchObject({ role: 'user', content: 'How do I run it?' });

    await act(async () => {
      resolveAsk({ answer: 'Run docker compose up.', sources: [] });
      await askPromise;
    });

    expect(result.current.pending).toBe(false);
    expect(result.current.turns).toHaveLength(2);
    expect(result.current.turns[1]).toMatchObject({
      role: 'assistant',
      content: 'Run docker compose up.',
    });
  });

  it('appends an error turn with the ApiClientError message on failure', async () => {
    askQuestionMock.mockRejectedValue(new ApiClientError('Re-ingest documents before querying.', 409));

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('alpha');
    });

    expect(result.current.pending).toBe(false);
    expect(result.current.turns[1]).toMatchObject({
      role: 'error',
      content: 'Re-ingest documents before querying.',
    });
  });

  it('appends a generic error turn for a non-ApiClientError failure', async () => {
    askQuestionMock.mockRejectedValue(new Error('boom'));

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
    let resolveAsk!: (value: { answer: string; sources: [] }) => void;
    askQuestionMock.mockReturnValue(
      new Promise((resolve) => {
        resolveAsk = resolve;
      }),
    );

    const { result } = renderHook(() => useChat());

    let firstAskPromise!: Promise<void>;
    act(() => {
      firstAskPromise = result.current.ask('first');
      void result.current.ask('second');
    });

    expect(askQuestionMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveAsk({ answer: 'ok', sources: [] });
      await firstAskPromise;
    });

    expect(result.current.pending).toBe(false);
    // Only the first question's user+assistant turns exist — the second call was a
    // no-op, not a second user turn queued behind it.
    expect(result.current.turns).toHaveLength(2);
  });

  it('allows a new ask() once the previous one has resolved', async () => {
    askQuestionMock.mockResolvedValueOnce({ answer: 'first answer', sources: [] });
    askQuestionMock.mockResolvedValueOnce({ answer: 'second answer', sources: [] });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('first');
    });
    await act(async () => {
      await result.current.ask('second');
    });

    expect(askQuestionMock).toHaveBeenCalledTimes(2);
    expect(result.current.turns).toHaveLength(4);
  });

  it('aborts the in-flight request when the component unmounts', async () => {
    askQuestionMock.mockImplementation(
      (_question, _provider, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    );

    const { result, unmount } = renderHook(() => useChat());

    act(() => {
      void result.current.ask('alpha');
    });

    const signal = askQuestionMock.mock.calls[0][2];
    expect(signal?.aborted).toBe(false);

    unmount();

    expect(signal?.aborted).toBe(true);
  });

  it('cancel() aborts the in-flight request and clears pending via the existing error path', async () => {
    askQuestionMock.mockImplementation(
      (_question, _provider, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new ApiClientError('Request timed out or was cancelled.')));
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
