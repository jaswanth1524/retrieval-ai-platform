import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BUSY_RETRY_ATTEMPTS,
  buildHistory,
  busyRetrySeconds,
  historyForQuestion,
  mergeConversations,
  useChat,
} from '../../src/hooks/useChat';
import { MAX_CONVERSATIONS, sanitizeTurn } from '../../src/hooks/chatPersistence';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { ApiClientError, api, setWorkspace } from '../../src/api/client';
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
  condense_ms: 0,
};

afterEach(() => {
  askQuestionStreamMock.mockReset();
  // useChat persists/restores turns via localStorage — clear between tests so one
  // test's turns don't leak into the next test's fresh renderHook().
  localStorage.clear();
});

describe('useChat', () => {
  it('marks an answer given from below-the-bar passages as a weak match', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('Foundation models are trained in house [1].', [], ZERO_TIMINGS, 't', false, {
          answerStatus: 'answered',
          topRerankScore: 0.006,
          weakMatches: true,
        });
      },
    );
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('what are foundational models');
    });

    const answer = result.current.turns[1];
    expect(answer.weakMatch).toBe(true);
    expect(answer.notFound).toBeUndefined();
    expect(sanitizeTurn(answer)?.weakMatch).toBe(true);
  });

  it('keeps a summary task on its turns and shows its part progress', async () => {
    const task = { mode: 'summarize' as const, filenames: ['a.pdf'] };
    let finish!: () => void;
    let fail!: () => void;
    askQuestionStreamMock.mockImplementation(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onStage?.('summarizing', { step: 2, steps: 3 });
        return new Promise<void>((resolve, reject) => {
          finish = () => {
            handlers.onDone?.('Sum [1]', [], ZERO_TIMINGS, null, false, {
              answerStatus: 'answered',
              coverage: { read: 4, total: 9 },
            });
            resolve();
          };
          fail = () => reject(new ApiClientError('boom'));
        });
      },
    );
    const { result } = renderHook(() => useChat());

    let asked!: Promise<void>;
    act(() => {
      asked = result.current.ask('Summarize a.pdf', undefined, undefined, undefined, { task });
    });
    expect(result.current.stage).toBe('taking notes, part 2 of 3…');
    expect(askQuestionStreamMock.mock.calls[0][7]).toMatchObject({ task });
    await act(async () => {
      finish();
      await asked;
    });
    expect(result.current.turns[0].task).toEqual(task);
    expect(result.current.turns[1].coverage).toEqual({ read: 4, total: 9 });

    act(() => {
      asked = result.current.ask('Summarize a.pdf', undefined, undefined, undefined, { task });
    });
    await act(async () => {
      fail();
      await asked;
    });
    const error = result.current.turns[result.current.turns.length - 1];
    expect(error.role).toBe('error');
    expect(error.task).toEqual(task);
  });

  it('streams sources then deltas into a single assistant turn, then finalizes on done', async () => {
    let resolveStream!: () => void;
    askQuestionStreamMock.mockImplementation((_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
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
    // Deltas are coalesced per animation frame rather than applied one render each.
    expect(result.current.turns[1]).toMatchObject({ role: 'assistant', content: '' });
    await act(async () => {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    });
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

  it('reflects the server stage frames, then generating, then clears', async () => {
    // Retrieval measured 25-80s against a real corpus while the UI showed one static
    // label for all of it. These are the real server frames driving a real indicator.
    let emit!: (fn: (h: QuestionStreamHandlers) => void) => void;
    let resolveStream!: () => void;
    askQuestionStreamMock.mockImplementation(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        emit = (fn) => fn(handlers);
        return new Promise<void>((resolve) => {
          resolveStream = () => {
            handlers.onDone?.('Answer [1].', [], ZERO_TIMINGS, 't');
            resolve();
          };
        });
      },
    );

    const { result } = renderHook(() => useChat());

    let askPromise!: Promise<void>;
    act(() => {
      askPromise = result.current.ask('What about the second one?');
    });

    // Seeded before any frame arrives, so there is never a pending request with no label.
    expect(result.current.stage).toBe('retrieving…');

    act(() => emit((h) => h.onStage?.('condensing')));
    expect(result.current.stage).toBe('reading the conversation…');

    act(() => emit((h) => h.onStage?.('searching')));
    expect(result.current.stage).toBe('searching documents…');

    // An unrecognised stage must degrade to the generic label, never surface raw.
    act(() => emit((h) => h.onStage?.('some-future-stage')));
    expect(result.current.stage).toBe('retrieving…');

    // Sources arriving is what marks the switch to generation — no server frame for it.
    act(() => emit((h) => h.onSources?.([], 't')));
    expect(result.current.stage).toBe('generating answer…');

    // After the last token, the server's zero-citation retry announces itself.
    act(() => emit((h) => h.onStage?.('verifying_citations')));
    expect(result.current.stage).toBe('checking citations…');

    await act(async () => {
      resolveStream();
      await askPromise;
    });

    expect(result.current.stage).toBeNull();
  });

  it('keeps the traceId set by the sources event even if the stream then errors', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
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

  it('marks a partly streamed answer incomplete when the stream then fails, and keeps it out of history', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onSources?.([], 'trace-3');
        handlers.onDelta?.('Alpha is half');
        throw new ApiClientError('The connection was lost.');
      },
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('alpha');
    });

    expect(result.current.turns[1]).toMatchObject({
      role: 'assistant',
      content: 'Alpha is half',
      incomplete: true,
    });
    expect(result.current.turns[2]).toMatchObject({ role: 'error', question: 'alpha' });
    // Its question goes too: alone it read to the model as an unanswered question.
    expect(buildHistory(result.current.turns)).toEqual([]);
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
      question: 'alpha',
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
    askQuestionStreamMock.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('first answer', [], ZERO_TIMINGS, 'trace-a');
    });
    askQuestionStreamMock.mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
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

    // The second call's history carries the first question and its answer.
    const secondCallHistory = askQuestionStreamMock.mock.calls[1][4];
    expect(secondCallHistory).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'first answer' },
    ]);
  });

  it('excludes error turns from the history sent with the next question', async () => {
    askQuestionStreamMock.mockRejectedValueOnce(new ApiClientError('boom'));
    askQuestionStreamMock.mockImplementationOnce(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('second answer', [], ZERO_TIMINGS, 'trace-b');
      },
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('first');
    });
    await act(async () => {
      await result.current.ask('second');
    });

    // The failed question goes too: sent alone it was two questions in a row.
    const secondCallHistory = askQuestionStreamMock.mock.calls[1][4];
    expect(secondCallHistory).toEqual([]);
  });

  it('aborts the in-flight request when the component unmounts', async () => {
    askQuestionStreamMock.mockImplementation(
      (_question, _provider, _overrides, _filenames, _history, _handlers, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    );

    const { result, unmount } = renderHook(() => useChat());

    act(() => {
      void result.current.ask('alpha');
    });

    const signal = askQuestionStreamMock.mock.calls[0][6];
    expect(signal?.aborted).toBe(false);

    unmount();

    expect(signal?.aborted).toBe(true);
  });

  it('cancel() clears pending and keeps the partial answer without an error turn', async () => {
    // A user pressing Stop deliberately ended their own request. Appending an error turn
    // under the partial answer would report a failure they caused on purpose.
    askQuestionStreamMock.mockImplementation(
      (_question, _provider, _overrides, _filenames, _history, handlers, signal) =>
        new Promise((_resolve, reject) => {
          handlers.onDelta?.('partial answer so far');
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
    expect(result.current.turns).toHaveLength(2);
    expect(result.current.turns[1]).toMatchObject({
      role: 'assistant',
      content: 'partial answer so far',
    });
    expect(result.current.turns.some((turn) => turn.role === 'error')).toBe(false);
  });

  it('cancel() before any delta leaves no empty assistant bubble behind', async () => {
    askQuestionStreamMock.mockImplementation(
      (_question, _provider, _overrides, _filenames, _history, handlers, signal) =>
        new Promise((_resolve, reject) => {
          // `sources` arrives first and creates the assistant turn, but no text follows.
          handlers.onSources?.([], 'trace-1');
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

    await act(async () => {
      result.current.cancel();
      await askPromise;
    });

    expect(result.current.turns).toHaveLength(1);
    expect(result.current.turns[0]).toMatchObject({ role: 'user', content: 'alpha' });
  });

  it('a timeout, unlike a user cancel, still produces an error turn', async () => {
    // A timeout aborts askQuestionStream's own private controller, never the one useChat
    // holds — so it must stay on the error path the cancel case above now skips.
    askQuestionStreamMock.mockRejectedValue(
      new ApiClientError('Request timed out or was cancelled.'),
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('alpha');
    });

    expect(result.current.turns[1]).toMatchObject({
      role: 'error',
      content: 'Request timed out or was cancelled.',
    });
  });

  it('cancel() is a no-op when nothing is in flight', () => {
    const { result } = renderHook(() => useChat());

    expect(() => result.current.cancel()).not.toThrow();
  });

  it('restores persisted turns on init', () => {
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 1,
        turns: [
          {
            id: 't1',
            role: 'user',
            content: 'earlier question',
            sources: [],
            timestamp: 0,
            timings: null,
            traceId: null,
          },
        ],
      }),
    );

    const { result } = renderHook(() => useChat());

    expect(result.current.turns).toHaveLength(1);
    expect(result.current.turns[0]).toMatchObject({ content: 'earlier question' });
  });

  it('ignores malformed persisted storage and starts with an empty thread', () => {
    localStorage.setItem('docrag-chat-history', 'not json');

    const { result } = renderHook(() => useChat());

    expect(result.current.turns).toEqual([]);
  });

  it('ignores persisted storage with a mismatched version', () => {
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({ version: 999, turns: [{ id: 't1', role: 'user', content: 'x' }] }),
    );

    const { result } = renderHook(() => useChat());

    expect(result.current.turns).toEqual([]);
  });

  it('persists turns to localStorage once a completed ask settles', async () => {
    askQuestionStreamMock.mockImplementationOnce(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('an answer', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('a question');
    });

    const raw = localStorage.getItem('docrag-chat-history');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed.version).toBe(2);
    expect(parsed.conversations).toHaveLength(1);
    expect(parsed.conversations[0].turns).toHaveLength(2);
    expect(parsed.conversations[0].turns[1]).toMatchObject({ content: 'an answer' });
  });

  it('migrates a v1 single-thread blob into the first conversation', () => {
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 1,
        turns: [
          { id: 't1', role: 'user', content: 'migrate me', sources: [], timestamp: 5, timings: null, traceId: null },
          { id: 't2', role: 'assistant', content: 'ok', sources: [], timestamp: 6, timings: null, traceId: null },
        ],
      }),
    );

    const { result } = renderHook(() => useChat());

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.turns).toHaveLength(2);
    expect(result.current.conversations[0].title).toBe('migrate me');
  });

  it('newConversation starts a fresh thread and switchConversation returns to the old one', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('answer one', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('first question');
    });
    const firstId = result.current.activeConversationId;

    act(() => {
      result.current.newConversation();
    });
    expect(result.current.turns).toEqual([]);
    expect(result.current.conversations.length).toBeGreaterThanOrEqual(2);

    act(() => {
      result.current.switchConversation(firstId!);
    });
    expect(result.current.turns).toHaveLength(2);
  });

  it('renameConversation and deleteConversation update the conversation list', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('answer', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('a question');
    });
    const id = result.current.activeConversationId!;

    act(() => {
      result.current.renameConversation(id, 'Renamed thread');
    });
    expect(result.current.conversations.find((c) => c.id === id)?.title).toBe('Renamed thread');

    act(() => {
      result.current.deleteConversation(id);
    });
    // Deleting the only conversation resets to a fresh empty one.
    expect(result.current.turns).toEqual([]);
    expect(result.current.conversations.find((c) => c.id === id)).toBeUndefined();
  });

  it('clear() empties turns and saves the emptied conversation', async () => {
    askQuestionStreamMock.mockImplementationOnce(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('an answer', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('a question');
    });
    expect(result.current.turns).toHaveLength(2);

    act(() => {
      result.current.clear();
    });

    expect(result.current.turns).toEqual([]);
    // Saved, not removed: other tabs have to learn the conversation was cleared.
    const persisted = JSON.parse(localStorage.getItem('docrag-chat-history') ?? 'null');
    expect(persisted.conversations).toHaveLength(1);
    expect(persisted.conversations[0].turns).toEqual([]);
  });

  it('clears the conversation in another tab too', async () => {
    const { result: mine } = renderHook(() => useChat());
    await act(async () => {
      await mine.current.ask('a question');
    });
    const beforeClear = localStorage.getItem('docrag-chat-history');
    expect(beforeClear).not.toBeNull();
    // A later millisecond than the answer, as any real clear is.
    await new Promise((resolve) => setTimeout(resolve, 5));

    act(() => mine.current.clear());
    const afterClear = localStorage.getItem('docrag-chat-history');

    // The other tab saw the conversation with its turns, then the clear.
    localStorage.setItem('docrag-chat-history', beforeClear as string);
    const { result: theirs } = renderHook(() => useChat());
    expect(theirs.current.turns.length).toBeGreaterThan(0);
    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', { key: 'docrag-chat-history', newValue: afterClear }),
      );
    });

    expect(theirs.current.turns).toEqual([]);
  });

  it('warns and resets to empty on an unparseable persisted blob', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    localStorage.setItem('docrag-chat-history', 'not json');

    const { result } = renderHook(() => useChat());

    expect(result.current.turns).toEqual([]);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('docrag-chat-history'));
    // Deliberately not copied to a sibling key: nothing reads such a copy back, so it
    // would double the footprint of a value that may be what exhausted the quota.
    expect(localStorage.getItem('docrag-chat-history.corrupt')).toBeNull();

    warn.mockRestore();
  });

  it('warns and resets to empty on a mismatched-version blob', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const raw = JSON.stringify({ version: 999, turns: [{ id: 't1', role: 'user', content: 'x' }] });
    localStorage.setItem('docrag-chat-history', raw);

    const { result } = renderHook(() => useChat());

    expect(result.current.turns).toEqual([]);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('docrag-chat-history'));
    expect(localStorage.getItem('docrag-chat-history.corrupt')).toBeNull();

    warn.mockRestore();
  });

  it('sets persistError and retries with a reduced payload when localStorage.setItem throws', async () => {
    askQuestionStreamMock.mockImplementationOnce(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('an answer', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const setItemSpy = vi
      .spyOn(Storage.prototype, 'setItem')
      .mockImplementationOnce(() => {
        throw new Error('QuotaExceededError');
      });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('a question');
    });

    expect(setItemSpy).toHaveBeenCalledTimes(2);
    expect(result.current.persistError).toBe(false);
    expect(localStorage.getItem('docrag-chat-history')).not.toBeNull();
    // The reduced write succeeded but dropped data from disk, so it must not report a
    // clean save — otherwise the loss only surfaces after a reload, with no warning.
    expect(result.current.persistPartial).toBe(true);

    setItemSpy.mockRestore();
  });

  it('reports persistPartial only until a full write succeeds again', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('an answer', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementationOnce(() => {
      throw new Error('QuotaExceededError');
    });

    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('a question');
    });
    expect(result.current.persistPartial).toBe(true);

    // Next turn persists normally: nothing is being dropped any more, so the warning
    // must clear rather than sticking for the rest of the session.
    await act(async () => {
      await result.current.ask('another question');
    });

    expect(result.current.persistPartial).toBe(false);
    expect(result.current.persistError).toBe(false);

    setItemSpy.mockRestore();
  });

  it('sets persistError when even the reduced-payload retry fails', async () => {
    askQuestionStreamMock.mockImplementationOnce(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('an answer', [], ZERO_TIMINGS, 'trace-1');
      },
    );

    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('a question');
    });

    expect(result.current.persistError).toBe(true);

    setItemSpy.mockRestore();
  });
});

describe('buildHistory', () => {
  function makeChatTurn(role: ChatTurn['role'], content: string): ChatTurn {
    return { id: role + content.length, role, content, sources: [], timestamp: 0, timings: null, traceId: null };
  }

  it('clips a summary answer harder than an ordinary one', () => {
    const summary = 'word '.repeat(1000);
    const history = buildHistory([
      { ...makeChatTurn('user', 'Summarize a.pdf'), task: { mode: 'summarize', filenames: ['a.pdf'] } },
      makeChatTurn('assistant', summary),
      makeChatTurn('user', 'Explain it'),
      makeChatTurn('assistant', summary),
    ]);

    expect(history[1].content.length).toBeLessThanOrEqual(1500);
    expect(history[1].content.endsWith('…')).toBe(true);
    expect(history[3].content.length).toBeGreaterThan(3500);
  });

  it('leaves out an exchange whose answer found nothing in the documents', () => {
    const history = buildHistory([
      makeChatTurn('user', 'How many vacation days?'),
      makeChatTurn('assistant', 'Twenty-eight [1].'),
      makeChatTurn('user', 'Explain the second point'),
      { ...makeChatTurn('assistant', 'I do not have enough information.'), notFound: true },
    ]);

    expect(history).toEqual([
      { role: 'user', content: 'How many vacation days?' },
      { role: 'assistant', content: 'Twenty-eight [1].' },
    ]);
  });

  it('truncates a single message within the per-message char cap', () => {
    // No whitespace anywhere, so the word-boundary backoff can't apply and this
    // exercises the hard-cut floor.
    const longAnswer = 'a'.repeat(5000);
    const history = buildHistory([
      makeChatTurn('user', 'question'),
      makeChatTurn('assistant', longAnswer),
    ]);

    expect(history).toHaveLength(2);
    // At most the cap, never over: the server rejects a longer message outright, so the
    // truncation marker has to fit inside the budget rather than be appended past it.
    expect(history[1].content.length).toBeLessThanOrEqual(4000);
    expect(history[1].content).toBe(`${'a'.repeat(3999)}…`);
  });

  it('backs off to a word boundary when one is close to the cap', () => {
    // A space just inside the cap: cutting at 3999 would sever "sever|ed", so the
    // truncation should retreat to the space instead.
    const head = 'word '.repeat(795); // 3975 chars, ends with a space
    const longAnswer = `${head}${'z'.repeat(200)}`;
    const history = buildHistory([
      makeChatTurn('user', 'question'),
      makeChatTurn('assistant', longAnswer),
    ]);

    expect(history[1].content.length).toBeLessThanOrEqual(4000);
    expect(history[1].content).toBe(`${head.trimEnd()}…`);
    // No partial word survived the cut.
    expect(history[1].content).not.toContain('z');
  });

  it('hard-cuts rather than backing off past the floor', () => {
    // The only space is far outside the backoff window, so honoring it would discard
    // most of the message; the hard cut has to win.
    const longAnswer = `a ${'b'.repeat(5000)}`;
    const history = buildHistory([
      makeChatTurn('user', 'question'),
      makeChatTurn('assistant', longAnswer),
    ]);

    expect(history[1].content.length).toBeLessThanOrEqual(4000);
    expect(history[1].content.startsWith('a b')).toBe(true);
    expect(history[1].content.endsWith('…')).toBe(true);
  });
});

describe('history for a repeated question', () => {
  function chatTurn(id: string, role: ChatTurn['role'], content: string): ChatTurn {
    return { id, role, content, sources: [], timestamp: 0, timings: null, traceId: null };
  }

  it('keeps only answered questions', () => {
    const turns = [
      chatTurn('1', 'user', 'first'),
      chatTurn('2', 'assistant', 'first answer'),
      chatTurn('3', 'user', 'failed one'),
      chatTurn('4', 'error', 'boom'),
      chatTurn('5', 'user', 'stopped one'),
    ];
    expect(buildHistory(turns)).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'first answer' },
    ]);
  });

  it('leaves out the exchange Regenerate repeats', () => {
    const turns = [
      chatTurn('1', 'user', 'first'),
      chatTurn('2', 'assistant', 'first answer'),
      chatTurn('3', 'user', 'second'),
      chatTurn('4', 'assistant', 'second answer'),
    ];
    expect(historyForQuestion(turns, 'second')).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'first answer' },
    ]);
    // A different question keeps everything.
    expect(historyForQuestion(turns, 'third')).toHaveLength(4);
  });

  it('never opens on an answer when the char budget splits a pair', () => {
    const turns = [
      chatTurn('1', 'user', 'q'.repeat(3999)),
      chatTurn('2', 'assistant', 'short'),
      chatTurn('3', 'user', 'last'),
      chatTurn('4', 'assistant', 'a'.repeat(4000)),
    ];
    // The budget runs out at the long question: its answer alone is left out too.
    expect(buildHistory(turns).map((message) => message.content)).toEqual([
      'last',
      'a'.repeat(4000),
    ]);
  });
});

describe('useChat storage robustness', () => {
  function savedTurn(id: string, role: ChatTurn['role'], content: string, sourceText = ''): ChatTurn {
    return {
      id,
      role,
      content,
      sources: sourceText
        ? [{ source_number: 1, filename: 'a.pdf', page: 1, section: 'S', chunk_id: 'c1', text: sourceText }]
        : [],
      timestamp: 1,
      timings: null,
      traceId: null,
    };
  }

  function seed(conversations: { id: string; updatedAt: number; turns?: ChatTurn[] }[]) {
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 2,
        activeConversationId: conversations[0].id,
        conversations: conversations.map((c) => ({
          title: c.id,
          createdAt: 0,
          turns: [savedTurn(`${c.id}-q`, 'user', `question ${c.id}`)],
          ...c,
        })),
      }),
    );
  }

  function saved(): { activeConversationId: string; conversations: { id: string; title: string; turns: ChatTurn[] }[] } {
    return JSON.parse(localStorage.getItem('docrag-chat-history') ?? 'null');
  }

  it('works where crypto.randomUUID is unavailable (plain HTTP on a LAN address)', () => {
    vi.stubGlobal('crypto', { getRandomValues: globalThis.crypto.getRandomValues.bind(globalThis.crypto) });
    try {
      const { result } = renderHook(() => useChat());
      expect(result.current.activeConversationId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('drops the least recently used conversation past the cap, not the oldest created', () => {
    // A full history; "daily" was created first but used most recently.
    const conversations = Array.from({ length: MAX_CONVERSATIONS }, (_, i) => ({
      id: `c${i}`,
      updatedAt: 100 + i,
    }));
    conversations[0] = { id: 'daily', updatedAt: 10_000 };
    seed(conversations);

    const { result } = renderHook(() => useChat());
    act(() => result.current.newConversation());

    const ids = result.current.conversations.map((c) => c.id);
    expect(ids).toContain('daily');
    expect(ids).not.toContain('c1');
    expect(ids).toHaveLength(MAX_CONVERSATIONS);
  });

  it('keeps passage text only on the most recent turns when saving', () => {
    const turns = Array.from({ length: 10 }, (_, i) => savedTurn(`t${i}`, 'assistant', `a${i}`, `passage ${i}`));
    seed([{ id: 'c', updatedAt: 1, turns }]);

    const { result } = renderHook(() => useChat());
    act(() => result.current.renameConversation('c', 'renamed'));

    const persisted = saved().conversations[0].turns;
    expect(persisted[0].sources[0]).toMatchObject({ chunk_id: 'c1', text: '' });
    expect(persisted[9].sources[0].text).toBe('passage 9');
    // In memory nothing is lost.
    expect(result.current.turns[0].sources[0].text).toBe('passage 0');
  });

  it("merges another tab's saved conversations instead of overwriting them", () => {
    seed([{ id: 'mine', updatedAt: 5 }]);
    const { result } = renderHook(() => useChat());

    const otherTab = JSON.stringify({
      version: 2,
      activeConversationId: 'theirs',
      conversations: [
        { id: 'theirs', title: 'theirs', createdAt: 0, updatedAt: 9, turns: [savedTurn('x', 'user', 'hi')] },
      ],
    });
    act(() => {
      window.dispatchEvent(new StorageEvent('storage', { key: 'docrag-chat-history', newValue: otherTab }));
    });

    expect(result.current.conversations.map((c) => c.id)).toEqual(['theirs', 'mine']);
    // This tab stays where it was.
    expect(result.current.activeConversationId).toBe('mine');
    expect(saved().conversations.map((c) => c.id).sort()).toEqual(['mine', 'theirs']);
  });

  it('saves an in-flight question when the page is hidden mid-answer', () => {
    askQuestionStreamMock.mockImplementation(() => new Promise<void>(() => {}));
    const { result } = renderHook(() => useChat());

    act(() => {
      void result.current.ask('Will this survive a reload?');
    });
    expect(localStorage.getItem('docrag-chat-history')).toBeNull();

    act(() => {
      window.dispatchEvent(new Event('pagehide'));
    });

    expect(saved().conversations[0].turns[0]).toMatchObject({
      role: 'user',
      content: 'Will this survive a reload?',
    });
  });

  it('saves a rename made in another conversation while an answer streams', () => {
    seed([
      { id: 'other', updatedAt: 1, turns: [savedTurn('o1', 'user', 'old question')] },
      { id: 'asking', updatedAt: 2 },
    ]);
    askQuestionStreamMock.mockImplementation(() => new Promise<void>(() => {}));
    const { result } = renderHook(() => useChat());
    act(() => result.current.switchConversation('asking'));
    act(() => {
      void result.current.ask('A long answer, please');
    });
    expect(result.current.pending).toBe(true);

    act(() => result.current.renameConversation('other', 'Renamed mid-answer'));

    // Saved while the answer is still streaming, not when it ends.
    expect(saved().conversations.find((c) => c.id === 'other')?.title).toBe('Renamed mid-answer');
    // The streaming answer itself is only saved as cut off, as on pagehide.
    expect(result.current.pending).toBe(true);
  });

  it('resets the title on Clear, so the next question names the conversation', () => {
    seed([{ id: 'c', updatedAt: 1 }]);
    const { result } = renderHook(() => useChat());

    act(() => result.current.clear());

    expect(result.current.conversations[0].title).toBe('New chat');
  });

  it('stores a feedback rating on its turn', () => {
    seed([{ id: 'c', updatedAt: 1, turns: [savedTurn('q', 'user', 'Q'), savedTurn('a', 'assistant', 'A')] }]);
    const { result } = renderHook(() => useChat());

    act(() => result.current.setTurnFeedback('a', 'up'));

    expect(result.current.turns[1].feedback).toBe('up');
    expect(saved().conversations[0].turns[1].feedback).toBe('up');
  });

  it('does not re-render the conversation list for streamed deltas', async () => {
    let handlers!: QuestionStreamHandlers;
    askQuestionStreamMock.mockImplementation((_q, _p, _o, _f, _h, h: QuestionStreamHandlers) => {
      handlers = h;
      return new Promise<void>(() => {});
    });
    const { result } = renderHook(() => useChat());
    act(() => {
      void result.current.ask('Q');
    });
    act(() => handlers.onSources?.([], null));
    const before = result.current.conversations;

    act(() => handlers.onDelta?.('more text'));
    await act(async () => {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    });

    expect(result.current.turns[1].content).toBe('more text');
    expect(result.current.conversations).toBe(before);
  });
});

describe('useChat feature pack', () => {
  function answeringStream(answer: string, cached = false) {
    askQuestionStreamMock.mockImplementation(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onSources?.([], null);
      handlers.onDone?.(answer, [], ZERO_TIMINGS, null, cached);
    });
  }

  it('passes question options through and marks cached answers', async () => {
    answeringStream('From cache.', true);
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('Q', 'ollama', undefined, ['a.pdf'], { tags: ['legal'], bypassCache: true });
    });

    const call = askQuestionStreamMock.mock.calls[0];
    expect(call[3]).toEqual(['a.pdf']);
    expect(call[7]).toEqual({ tags: ['legal'], bypassCache: true });
    expect(result.current.turns[1]).toMatchObject({ content: 'From cache.', cached: true });
  });

  it('flags an answer the server says the documents do not contain, and keeps the flag', async () => {
    askQuestionStreamMock.mockImplementation(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onSources?.([], null);
      handlers.onDone?.('Not in the documents.', [], ZERO_TIMINGS, null, false, {
        answerStatus: 'insufficient_context',
        topRerankScore: 0.2,
      });
    });
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('Q', 'ollama');
    });

    expect(result.current.turns[1]).toMatchObject({ content: 'Not in the documents.', notFound: true });
    expect(sanitizeTurn(JSON.parse(JSON.stringify(result.current.turns[1])))?.notFound).toBe(true);
  });

  it('flags an uncited or cut-off answer, and keeps both flags', async () => {
    askQuestionStreamMock.mockImplementation(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Alpha is beta and', [], ZERO_TIMINGS, null, false, {
        answerStatus: 'uncited',
        truncated: true,
      });
    });
    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.ask('Q', 'ollama');
    });

    expect(result.current.turns[1]).toMatchObject({ uncited: true, truncated: true });
    expect(result.current.turns[1].notFound).toBeUndefined();
    const restored = sanitizeTurn(JSON.parse(JSON.stringify(result.current.turns[1])));
    expect(restored).toMatchObject({ uncited: true, truncated: true });
  });

  it('does not flag an answered question', async () => {
    askQuestionStreamMock.mockImplementation(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Yes [1].', [], ZERO_TIMINGS, null, false, { answerStatus: 'answered' });
    });
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('Q', 'ollama');
    });
    expect(result.current.turns[1].notFound).toBeUndefined();
  });

  it('pins up to the limit, and finds conversations by their answers', async () => {
    answeringStream('The travel policy allows economy fares.');
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('Which fares?', 'ollama');
    });
    const id = result.current.activeConversationId as string;

    let pinned = false;
    act(() => {
      pinned = result.current.setPinned(id, true);
    });
    expect(pinned).toBe(true);
    expect(result.current.conversations.find((c) => c.id === id)?.pinned).toBe(true);

    const match = result.current.searchConversations('economy').get(id);
    expect(match?.snippet).toContain('allows economy fares');
    expect(result.current.turns.find((turn) => turn.id === match?.turnId)?.role).toBe('assistant');
  });

  it('edits a question into a new conversation, leaving the original intact', async () => {
    answeringStream('First answer.');
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('First question');
    });
    const originalId = result.current.activeConversationId;
    const questionId = result.current.turns[0].id;

    answeringStream('Edited answer.');
    await act(async () => {
      await result.current.editAndResend(questionId, 'Edited question');
    });

    expect(result.current.activeConversationId).not.toBe(originalId);
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['Edited question', 'Edited answer.']);
    // Forked before the first question: no history rides along.
    expect(askQuestionStreamMock.mock.calls[1][4]).toEqual([]);
    act(() => result.current.switchConversation(originalId!));
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['First question', 'First answer.']);
  });

  it('marks a stopped partial answer and leaves it out of the next history', async () => {
    askQuestionStreamMock.mockImplementation(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers, signal?: AbortSignal) =>
        new Promise<void>((_resolve, reject) => {
          handlers.onSources?.([], null);
          handlers.onDelta?.('Partial');
          signal?.addEventListener('abort', () => reject(new ApiClientError('cancelled')));
        }),
    );
    const { result } = renderHook(() => useChat());
    let pendingAsk!: Promise<void>;
    act(() => {
      pendingAsk = result.current.ask('Q');
    });
    await act(async () => {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    });
    await act(async () => {
      result.current.cancel();
      await pendingAsk;
    });

    expect(result.current.turns[1]).toMatchObject({ content: 'Partial', stopped: true });
    expect(buildHistory(result.current.turns)).toEqual([]);
  });

  it('imports a JSON export as a new conversation with fresh ids', () => {
    const { result } = renderHook(() => useChat());
    const exported: ChatTurn[] = [
      { id: 'x', role: 'user', content: 'Imported Q', sources: [], timestamp: 1, timings: null, traceId: null },
      { id: 'y', role: 'assistant', content: 'Imported A', sources: [], timestamp: 2, timings: null, traceId: null },
    ];

    let ok = false;
    act(() => {
      ok = result.current.importConversation(exported);
    });

    expect(ok).toBe(true);
    expect(result.current.conversations[0].title).toBe('Imported Q');
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['Imported Q', 'Imported A']);
    expect(result.current.turns[0].id).not.toBe('x');
    expect(result.current.importConversation({ nope: true })).toBe(false);
    expect(result.current.importConversation([{ role: 'bogus' }])).toBe(false);
  });

  it('keeps writing an answer to its own conversation while others are opened', async () => {
    let finish!: () => void;
    askQuestionStreamMock.mockImplementation(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) =>
        new Promise<void>((resolve) => {
          handlers.onDelta?.('Part');
          finish = () => {
            handlers.onDone?.('Whole answer.', [], ZERO_TIMINGS, null);
            resolve();
          };
        }),
    );
    const { result } = renderHook(() => useChat());
    const askedIn = result.current.activeConversationId!;
    let asking!: Promise<void>;
    act(() => {
      asking = result.current.ask('Q');
    });
    expect(result.current.streamingConversationId).toBe(askedIn);

    // A new conversation, an import and a switch all go through mid-answer.
    act(() => result.current.newConversation());
    const fresh = result.current.activeConversationId!;
    expect(fresh).not.toBe(askedIn);
    let imported = false;
    act(() => {
      imported = result.current.importConversation([
        { id: 'x', role: 'user', content: 'Imported', sources: [], timestamp: 1, timings: null, traceId: null },
      ]);
    });
    expect(imported).toBe(true);
    // The conversation being answered can't be deleted under the answer; others can.
    act(() => result.current.deleteConversation(askedIn));
    expect(result.current.conversations.some((c) => c.id === askedIn)).toBe(true);
    act(() => result.current.deleteConversation(fresh));
    expect(result.current.conversations.some((c) => c.id === fresh)).toBe(false);

    await act(async () => {
      finish();
      await asking;
    });

    expect(result.current.streamingConversationId).toBeNull();
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['Imported']);
    expect(result.current.askedTurns.map((turn) => turn.content)).toEqual(['Q', 'Whole answer.']);
    act(() => result.current.switchConversation(askedIn));
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['Q', 'Whole answer.']);
  });

  it('keeps a search scope per conversation and prunes deleted documents from all of them', () => {
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: ['a.pdf', 'b.pdf'], tags: ['legal'] }));
    const scopedId = result.current.activeConversationId!;
    // A new conversation needs a turn in the scoped one first (empty ones are reused).
    act(() => result.current.importConversation([
      { id: 'q', role: 'user', content: 'Other', sources: [], timestamp: 1, timings: null, traceId: null },
    ]));

    expect(result.current.scope).toEqual({ filenames: [], tags: [] });

    act(() => result.current.pruneScopes(['b.pdf']));
    act(() => result.current.switchConversation(scopedId));
    expect(result.current.scope).toEqual({ filenames: ['b.pdf'], tags: ['legal'] });
  });

  it('a new chat that reuses an empty conversation drops the scope set on it', () => {
    // "Ask about these" scoped the empty conversation; New chat then kept it, and the
    // first question of the "fresh" chat was searched in those documents only.
    const { result } = renderHook(() => useChat());
    const id = result.current.activeConversationId;
    act(() => result.current.setScope({ filenames: ['a.pdf'], tags: ['legal'] }));

    act(() => result.current.newConversation());

    expect(result.current.activeConversationId).toBe(id);
    expect(result.current.scope).toEqual({ filenames: [], tags: [] });
  });

  it('prunes tags no document carries any more, ignoring case', () => {
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: [], tags: ['Legal', 'hr'] }));

    act(() => result.current.pruneScopes([], ['legal', 'finance']));
    expect(result.current.scope).toEqual({ filenames: [], tags: ['Legal'] });

    // A listing without tags (an older server) leaves them alone.
    act(() => result.current.pruneScopes([]));
    expect(result.current.scope.tags).toEqual(['Legal']);
  });
});

describe('useChat scope updates', () => {
  it('removes only the named documents from scopes, without a corpus snapshot', () => {
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: ['a.pdf', 'new.pdf'], tags: [] }));

    // 'new.pdf' finished uploading while the delete of 'a.pdf' was in flight; a prune
    // against the pre-delete corpus list would have dropped it too.
    act(() => result.current.removeFromScopes(['a.pdf']));

    expect(result.current.scope.filenames).toEqual(['new.pdf']);
  });

  it('applies two scope updaters in one tick without one undoing the other', () => {
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: ['a.pdf'], tags: ['legal'] }));

    // What "All documents" does: clear the files, then the tags, back to back.
    act(() => {
      result.current.setScope((prev) => ({ ...prev, filenames: [] }));
      result.current.setScope((prev) => ({ ...prev, tags: [] }));
    });

    expect(result.current.scope).toEqual({ filenames: [], tags: [] });
  });
});

describe('useChat reload mid-answer', () => {
  it('saves a partial answer as stopped when the page is hidden mid-stream', async () => {
    let handlers!: QuestionStreamHandlers;
    askQuestionStreamMock.mockImplementation((_q, _p, _o, _f, _h, h: QuestionStreamHandlers) => {
      handlers = h;
      return new Promise<void>(() => {});
    });
    const { result } = renderHook(() => useChat());
    act(() => {
      void result.current.ask('Q');
    });
    act(() => handlers.onSources?.([], null));
    act(() => handlers.onDelta?.('Half an ans'));
    await act(async () => {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    });

    act(() => {
      window.dispatchEvent(new Event('pagehide'));
    });

    const saved = JSON.parse(localStorage.getItem('docrag-chat-history') ?? 'null');
    expect(saved.conversations[0].turns[1]).toMatchObject({ content: 'Half an ans', stopped: true });
    // In memory the answer is still streaming, not stopped.
    expect(result.current.turns[1].stopped).toBeUndefined();
  });

  it('saves the question and unflushed text when the tab is hidden mid-stream', () => {
    // A hidden tab runs no animation frames, and a mobile browser can discard it with
    // no pagehide: the question and the answer so far were never written.
    let handlers!: QuestionStreamHandlers;
    askQuestionStreamMock.mockImplementation((_q, _p, _o, _f, _h, h: QuestionStreamHandlers) => {
      handlers = h;
      return new Promise<void>(() => {});
    });
    const { result } = renderHook(() => useChat());
    act(() => {
      void result.current.ask('Q');
    });
    act(() => handlers.onSources?.([], null));
    act(() => handlers.onDelta?.('Not yet painted'));

    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    try {
      act(() => {
        document.dispatchEvent(new Event('visibilitychange'));
      });
    } finally {
      visibility.mockRestore();
    }

    const saved = JSON.parse(localStorage.getItem('docrag-chat-history') ?? 'null');
    expect(saved.conversations[0].turns[0]).toMatchObject({ role: 'user', content: 'Q' });
    expect(saved.conversations[0].turns[1]).toMatchObject({
      content: 'Not yet painted',
      stopped: true,
    });
    expect(result.current.turns[1]?.stopped).toBeUndefined();
  });
});

describe('useChat cross-tab changes mid-answer', () => {
  function startStream() {
    let handlers!: QuestionStreamHandlers;
    let finish!: () => void;
    askQuestionStreamMock.mockImplementation((_q, _p, _o, _f, _h, h: QuestionStreamHandlers) => {
      handlers = h;
      return new Promise<void>((resolve) => {
        finish = resolve;
      });
    });
    const hook = renderHook(() => useChat());
    let pendingAsk!: Promise<void>;
    act(() => {
      pendingAsk = hook.result.current.ask('Q');
    });
    act(() => handlers.onSources?.([], 'trace-1'));
    const conversationId = hook.result.current.activeConversationId!;
    return {
      hook,
      conversationId,
      handlers: () => handlers,
      finish: async () => {
        await act(async () => {
          finish();
          await pendingAsk;
        });
      },
    };
  }

  function otherTabWrites(payload: unknown) {
    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', {
          key: 'docrag-chat-history',
          newValue: JSON.stringify(payload),
        }),
      );
    });
  }

  it("takes another tab's rename but keeps the streaming answer", async () => {
    const { hook, conversationId, handlers, finish } = startStream();
    const later = Date.now() + 60_000;
    otherTabWrites({
      version: 2,
      activeConversationId: conversationId,
      conversations: [
        { id: conversationId, title: 'Renamed', createdAt: 0, updatedAt: 0, editedAt: later, turns: [] },
      ],
    });
    act(() => handlers().onDone?.('Full answer', [], ZERO_TIMINGS, 'trace-1'));
    await finish();

    const { result } = hook;
    expect(result.current.activeConversationId).toBe(conversationId);
    expect(result.current.conversations[0].title).toBe('Renamed');
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['Q', 'Full answer']);
  });

  it("applies another tab's delete once the answer has finished streaming", async () => {
    const { hook, conversationId, handlers, finish } = startStream();
    otherTabWrites({
      version: 2,
      activeConversationId: 'other',
      conversations: [
        {
          id: 'other',
          title: 'Other',
          createdAt: 0,
          updatedAt: Date.now() + 60_000,
          turns: [
            { id: 'o1', role: 'user', content: 'other q', sources: [], timestamp: 0, timings: null, traceId: null },
          ],
        },
      ],
      deleted: [{ id: conversationId, at: Date.now() + 60_000 }],
    });
    act(() => handlers().onDone?.('Full answer', [], ZERO_TIMINGS, 'trace-1'));
    await finish();

    const { result } = hook;
    // The stream finished into its own conversation, never into 'other' ...
    expect(result.current.conversations.map((c) => c.id)).toEqual(['other']);
    expect(result.current.activeConversationId).toBe('other');
    expect(result.current.turns.map((turn) => turn.content)).toEqual(['other q']);
    // ... and the delete then won: what this tab saves keeps it deleted for the other tab.
    const saved = JSON.parse(window.localStorage.getItem('docrag-chat-history') ?? '{}');
    expect(saved.conversations.map((c: { id: string }) => c.id)).toEqual(['other']);
    expect(saved.deleted.map((d: { id: string }) => d.id)).toContain(conversationId);
  });

  it('keeps the streaming conversation visible until the answer is done', () => {
    const { hook, conversationId } = startStream();
    otherTabWrites({
      version: 2,
      activeConversationId: conversationId,
      conversations: [],
      deleted: [{ id: conversationId, at: Date.now() + 60_000 }],
    });

    expect(hook.result.current.activeConversationId).toBe(conversationId);
    expect(hook.result.current.conversations.map((c) => c.id)).toContain(conversationId);
    expect(hook.result.current.turns[0].content).toBe('Q');
  });
});

describe('mergeConversations', () => {
  it("keeps this tab's passage text when another tab's compact copy is newer", () => {
    const source = (text: string) => ({
      source_number: 1,
      filename: 'a.pdf',
      page: 1,
      section: 'S',
      chunk_id: 'c',
      text,
    });
    const answer = (text: string): ChatTurn => ({
      id: 'a',
      role: 'assistant',
      content: 'A',
      sources: [source(text)],
      timestamp: 1,
      timings: null,
      traceId: null,
    });
    const local = {
      version: 2 as const,
      activeConversationId: 'c1',
      conversations: [{ id: 'c1', title: 'T', createdAt: 1, updatedAt: 5, turns: [answer('full text')] }],
    };
    const fromOtherTab = [{ id: 'c1', title: 'T', createdAt: 1, updatedAt: 9, turns: [answer('')] }];

    const merged = mergeConversations(local, fromOtherTab);

    expect(merged.conversations[0].updatedAt).toBe(9);
    expect(merged.conversations[0].turns[0].sources[0].text).toBe('full text');
  });
});

describe('useChat workspaces', () => {
  afterEach(() => setWorkspace(''));

  const doneStream = () =>
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('Answer [1]', [], ZERO_TIMINGS, null, false);
      },
    );

  it('binds a conversation to the workspace of its first question', async () => {
    doneStream();
    setWorkspace('legal');
    const { result } = renderHook(() => useChat());
    expect(result.current.conversationWorkspace).toBeNull();

    await act(async () => {
      await result.current.ask('Q1');
    });
    expect(result.current.conversationWorkspace).toBe('legal');
    expect(result.current.conversations[0].workspace).toBe('legal');

    // Later questions don't move it, even from another workspace.
    setWorkspace('');
    await act(async () => {
      await result.current.ask('Q2');
    });
    expect(result.current.conversationWorkspace).toBe('legal');
  });

  it('records the default workspace as no field, and keeps the binding on an edited fork', async () => {
    doneStream();
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('Q1');
    });
    expect(result.current.conversationWorkspace).toBe('');
    expect(result.current.conversations[0].workspace).toBeUndefined();

    setWorkspace('legal');
    const bound = renderHook(() => useChat());
    act(() => bound.result.current.newConversation());
    await act(async () => {
      await bound.result.current.ask('Q2');
    });
    const userTurn = bound.result.current.turns[0];
    await act(async () => {
      await bound.result.current.editAndResend(userTurn.id, 'Q2 edited');
    });
    expect(bound.result.current.conversationWorkspace).toBe('legal');
  });

  it("prunes scopes only in the current workspace's conversations", async () => {
    doneStream();
    setWorkspace('legal');
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: ['contract.pdf'], tags: [] }));
    await act(async () => {
      await result.current.ask('Q');
    });
    const legalId = result.current.activeConversationId!;

    // Back in the default workspace, whose listing has no contract.pdf.
    setWorkspace('');
    act(() => result.current.newConversation());
    act(() => result.current.pruneScopes(['guide.md']));
    act(() => result.current.switchConversation(legalId));
    expect(result.current.scope.filenames).toEqual(['contract.pdf']);
  });

  it('imports a conversation into the workspace in use', () => {
    setWorkspace('legal');
    const { result } = renderHook(() => useChat());
    act(() => {
      result.current.importConversation([
        { id: 'q', role: 'user', content: 'Q', sources: [], timestamp: 1, timings: null, traceId: null },
      ]);
    });
    expect(result.current.conversationWorkspace).toBe('legal');
  });
});

describe('useChat removeFromScopes', () => {
  afterEach(() => setWorkspace(''));

  it("prunes the conversations of the workspace a document was deleted from", async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('Answer', [], ZERO_TIMINGS, null, false);
      },
    );
    setWorkspace('legal');
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: ['shared.md'], tags: [] }));
    await act(async () => {
      await result.current.ask('Q');
    });

    // The delete was started in legal and ends after a switch to the default workspace.
    setWorkspace('');
    act(() => result.current.removeFromScopes(['shared.md'], 'legal'));
    expect(result.current.scope.filenames).toEqual([]);
  });

  it("leaves a fresh chat of the workspace in use alone when another workspace's delete ends", () => {
    // No questions yet, so bound to no workspace: it belongs to the default one in use,
    // whose own shared.md is still there.
    const { result } = renderHook(() => useChat());
    act(() => result.current.setScope({ filenames: ['shared.md'], tags: [] }));

    act(() => result.current.removeFromScopes(['shared.md'], 'legal'));
    expect(result.current.scope.filenames).toEqual(['shared.md']);

    act(() => result.current.removeFromScopes(['shared.md']));
    expect(result.current.scope.filenames).toEqual([]);
  });
});

describe('useChat follow-up suggestions', () => {
  it('finishes the answer at done, and attaches suggestions arriving later to that answer', async () => {
    let sendFollowups!: (questions: string[]) => void;
    let closeStream!: () => void;
    askQuestionStreamMock.mockImplementationOnce(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onSources?.([], 'trace-1');
        handlers.onDone?.('First answer.', [], ZERO_TIMINGS, 'trace-1');
        sendFollowups = (questions) => handlers.onFollowups?.(questions);
        return new Promise<void>((resolve) => (closeStream = resolve));
      },
    );
    let finishSecond!: () => void;
    askQuestionStreamMock.mockImplementationOnce(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) =>
        new Promise<void>((resolve) => {
          finishSecond = () => {
            handlers.onDone?.('Second answer.', [], ZERO_TIMINGS, 'trace-2');
            resolve();
          };
        }),
    );
    const { result } = renderHook(() => useChat());

    let first!: Promise<void>;
    act(() => {
      first = result.current.ask('First?');
    });
    // Done, though the stream is still open for suggestions: not streaming any more.
    expect(result.current.pending).toBe(false);
    expect(result.current.streamingConversationId).toBeNull();
    expect(result.current.stage).toBeNull();
    const firstAnswerId = result.current.turns[1].id;

    act(() => sendFollowups(['Why?', 'Why?', '  ', 'How?']));
    expect(result.current.turns[1].followups).toEqual(['Why?', 'How?']);

    // The next question goes out at once, not after the suggestions stream closes.
    let second!: Promise<void>;
    act(() => {
      second = result.current.ask('Second?');
    });
    expect(askQuestionStreamMock).toHaveBeenCalledTimes(2);
    expect(result.current.pending).toBe(true);
    // ...and stops waiting for suggestions to an answer that is no longer the last.
    expect((askQuestionStreamMock.mock.calls[0][6] as AbortSignal).aborted).toBe(true);

    // A late suggestion lands on its own answer, never on the one streaming now; the
    // first stream ending doesn't end the second question.
    act(() => sendFollowups(['Late?']));
    await act(async () => {
      closeStream();
      await first;
    });
    expect(result.current.pending).toBe(true);
    expect(result.current.turns.find((turn) => turn.id === firstAnswerId)?.followups).toEqual(['Late?']);
    expect(result.current.turns.filter((turn) => turn.followups)).toHaveLength(1);

    await act(async () => {
      finishSecond();
      await second;
    });
    expect(result.current.pending).toBe(false);
    expect(result.current.turns[3]).toMatchObject({ role: 'assistant', content: 'Second answer.' });
  });

  it('Stop after done leaves the finished answer alone', async () => {
    askQuestionStreamMock.mockImplementationOnce(
      (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers, signal?: AbortSignal) => {
        handlers.onDone?.('First answer.', [], ZERO_TIMINGS, null);
        // Still open for suggestions; an abort would end it as a cancellation.
        return new Promise<void>((_resolve, reject) =>
          signal?.addEventListener('abort', () => reject(new ApiClientError('cancelled'))),
        );
      },
    );
    const { result } = renderHook(() => useChat());
    act(() => {
      void result.current.ask('First?');
    });
    await act(async () => {
      result.current.cancel();
      await Promise.resolve();
    });
    expect(result.current.turns[1]).toMatchObject({ content: 'First answer.' });
    expect(result.current.turns[1].stopped).toBeUndefined();
    expect((askQuestionStreamMock.mock.calls[0][6] as AbortSignal).aborted).toBe(false);
  });
});

describe('useChat follow-up flag (feedback kind)', () => {
  it('marks an answer asked with the conversation as history, not the first or a summary', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('An answer.', [], ZERO_TIMINGS, null);
      },
    );
    const { result } = renderHook(() => useChat());
    await act(() => result.current.ask('First?'));
    await act(() => result.current.ask('Then?'));
    await act(() =>
      result.current.ask('Summarize a.pdf', undefined, undefined, undefined, {
        task: { mode: 'summarize', filenames: ['a.pdf'] },
      }),
    );

    const answers = result.current.turns.filter((turn) => turn.role === 'assistant');
    expect(answers.map((turn) => Boolean(turn.followUp))).toEqual([false, true, false]);
    // Kept across a reload.
    expect(sanitizeTurn(answers[1])?.followUp).toBe(true);
  });
});

describe('pass 7', () => {
  it('waits out a busy server and asks again on its own', async () => {
    vi.useFakeTimers();
    try {
      askQuestionStreamMock
        .mockRejectedValueOnce(new ApiClientError('Busy', 429, 2))
        .mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
          handlers.onDone?.('Ten days [1].', [], ZERO_TIMINGS, 't1', false, {
            topRerankScore: 0.42,
            usage: { prompt_tokens: 100, completion_tokens: 20 },
          });
        });
      const { result } = renderHook(() => useChat());
      let asked!: Promise<void>;
      act(() => {
        asked = result.current.ask('Leave?');
      });
      await act(async () => {
        await Promise.resolve();
      });
      expect(result.current.stage).toMatch(/asking again in 2s/);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
        await asked;
      });
      expect(askQuestionStreamMock).toHaveBeenCalledTimes(2);
      const answer = result.current.turns.find((turn) => turn.role === 'assistant');
      expect(answer?.content).toBe('Ten days [1].');
      expect(answer?.topRerankScore).toBe(0.42);
      expect(answer?.usage).toEqual({ prompt: 100, completion: 20 });
      expect(result.current.turns.some((turn) => turn.role === 'error')).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('gives up after a few refusals, as an error turn', () => {
    expect(busyRetrySeconds(new ApiClientError('Busy', 429, 5), 1)).toBe(5);
    expect(busyRetrySeconds(new ApiClientError('Busy', 429), 1)).toBe(5);
    expect(busyRetrySeconds(new ApiClientError('Busy', 429, 999), 1)).toBe(60);
    expect(busyRetrySeconds(new ApiClientError('Busy', 429, 5), BUSY_RETRY_ATTEMPTS + 1)).toBeNull();
    expect(busyRetrySeconds(new ApiClientError('Gone', 404), 1)).toBeNull();
  });

  it('marks a source not relevant, and exports every conversation for import', async () => {
    askQuestionStreamMock.mockImplementation(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('Answer [1].', [], ZERO_TIMINGS, 't1', false);
      },
    );
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('First?');
    });
    const answer = result.current.turns.find((turn) => turn.role === 'assistant')!;
    act(() => result.current.setSourceIrrelevant(answer.id, 2, true));
    expect(result.current.turns.find((turn) => turn.id === answer.id)?.irrelevantSources).toEqual([2]);
    act(() => result.current.setSourceIrrelevant(answer.id, 2, false));
    expect(
      result.current.turns.find((turn) => turn.id === answer.id)?.irrelevantSources,
    ).toBeUndefined();

    const exported = JSON.parse(result.current.allConversationsJson());
    expect(exported.docrag_conversations).toHaveLength(1);
    const [item] = exported.docrag_conversations;
    act(() => {
      result.current.importConversation(item, item.title);
    });
    expect(result.current.conversations.filter((c) => c.turnCount > 0)).toHaveLength(2);
  });
});

it('asking an unanswered question again replaces it rather than repeating it', async () => {
  askQuestionStreamMock.mockImplementationOnce(
    (_q, _p, _o, _f, _h, _handlers, signal?: AbortSignal) =>
      new Promise<void>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new ApiClientError('Cancelled.')));
      }),
  );
  const { result } = renderHook(() => useChat());
  let first!: Promise<void>;
  act(() => {
    first = result.current.ask('Leave?');
  });
  await act(async () => {
    result.current.cancel();
    await first;
  });
  expect(result.current.turns.map((turn) => turn.role)).toEqual(['user']);

  askQuestionStreamMock.mockImplementationOnce(
    async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Ten days [1].', [], ZERO_TIMINGS, 't1', false);
    },
  );
  await act(async () => {
    await result.current.ask('Leave?');
  });
  expect(result.current.turns.map((turn) => [turn.role, turn.content])).toEqual([
    ['user', 'Leave?'],
    ['assistant', 'Ten days [1].'],
  ]);
});

it('imports a whole export into the room left, keeping its dates and every existing chat', () => {
  const { result } = renderHook(() => useChat());
  const turn = (id: string, role: string, content: string) => ({
    id,
    role,
    content,
    sources: [],
    timestamp: 1,
    timings: null,
    traceId: null,
  });
  const pair = (n: number) => [
    turn(`q${n}`, 'user', `Question ${n}?`),
    turn(`a${n}`, 'assistant', `Answer ${n}.`),
  ];
  act(() => {
    for (let n = 0; n < MAX_CONVERSATIONS - 3; n += 1) result.current.importConversation(pair(n));
  });
  const before = new Set(result.current.conversations.map((c) => c.id));
  const old = Date.UTC(2025, 0, 1);
  const items = [1, 2, 3, 4, 5].map((n) => ({
    raw: { turns: pair(100 + n), createdAt: old + n, updatedAt: old + n * 1000 },
    title: `Imported ${n}`,
  }));

  let outcome!: { imported: number; skipped: number };
  act(() => {
    outcome = result.current.importConversations(items);
  });

  expect(outcome.imported + outcome.skipped).toBe(5);
  expect(outcome.skipped).toBeGreaterThan(0);
  const after = result.current.conversations;
  expect(after.length).toBeLessThanOrEqual(MAX_CONVERSATIONS);
  // Nothing that was there is gone.
  for (const id of before) expect(after.some((c) => c.id === id)).toBe(true);
  // The newest of the imported ones came in, with its own date.
  const newest = after.find((c) => c.title === 'Imported 5');
  expect(newest?.updatedAt).toBe(old + 5000);
});

it('makes room for an import only when asked, never removing a pinned or the active chat', () => {
  const { result } = renderHook(() => useChat());
  const turn = (id: string, role: string, content: string) => ({
    id,
    role,
    content,
    sources: [],
    timestamp: 1,
    timings: null,
    traceId: null,
  });
  const pair = (n: number) => [turn(`q${n}`, 'user', `Q ${n}?`), turn(`a${n}`, 'assistant', `A ${n}.`)];
  act(() => {
    for (let n = 0; n < MAX_CONVERSATIONS - 2; n += 1) result.current.importConversation(pair(n));
  });
  // The oldest one pinned: it stays whatever happens.
  const oldest = [...result.current.conversations].sort((a, b) => a.updatedAt - b.updatedAt)[0];
  act(() => {
    result.current.setPinned(oldest.id, true);
  });
  const now = Date.now();
  const items = [1, 2, 3, 4, 5].map((n) => ({
    raw: { turns: pair(500 + n), createdAt: now - n, updatedAt: now - n },
  }));
  expect(result.current.importPreview(items)).toEqual({ valid: 5, room: 1 });

  let outcome!: { imported: number; skipped: number; removed: number };
  act(() => {
    outcome = result.current.importConversations(items, { makeRoom: true });
  });

  expect(outcome).toEqual({ imported: 5, skipped: 0, removed: 4 });
  expect(result.current.conversations.length).toBe(MAX_CONVERSATIONS);
  expect(result.current.conversations.some((c) => c.id === oldest.id)).toBe(true);
});
