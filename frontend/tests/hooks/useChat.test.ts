import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildHistory, mergeConversations, useChat } from '../../src/hooks/useChat';
import type { ChatTurn } from '../../src/components/ChatMessage';
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
  condense_ms: 0,
};

afterEach(() => {
  askQuestionStreamMock.mockReset();
  // useChat persists/restores turns via localStorage — clear between tests so one
  // test's turns don't leak into the next test's fresh renderHook().
  localStorage.clear();
});

describe('useChat', () => {
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

    const secondCallHistory = askQuestionStreamMock.mock.calls[1][4];
    expect(secondCallHistory).toEqual([{ role: 'user', content: 'first' }]);
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

  it('clear() empties turns and removes the persisted storage entry', async () => {
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
    expect(localStorage.getItem('docrag-chat-history')).toBeNull();
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
    const history = buildHistory([makeChatTurn('assistant', longAnswer)]);

    expect(history[0].content.length).toBeLessThanOrEqual(4000);
    expect(history[0].content).toBe(`${head.trimEnd()}…`);
    // No partial word survived the cut.
    expect(history[0].content).not.toContain('z');
  });

  it('hard-cuts rather than backing off past the floor', () => {
    // The only space is far outside the backoff window, so honoring it would discard
    // most of the message; the hard cut has to win.
    const longAnswer = `a ${'b'.repeat(5000)}`;
    const history = buildHistory([makeChatTurn('assistant', longAnswer)]);

    expect(history[0].content.length).toBeLessThanOrEqual(4000);
    expect(history[0].content.startsWith('a b')).toBe(true);
    expect(history[0].content.endsWith('…')).toBe(true);
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
    // 20 conversations; "daily" was created first but used most recently.
    const conversations = Array.from({ length: 20 }, (_, i) => ({ id: `c${i}`, updatedAt: 100 + i }));
    conversations[0] = { id: 'daily', updatedAt: 10_000 };
    seed(conversations);

    const { result } = renderHook(() => useChat());
    act(() => result.current.newConversation());

    const ids = result.current.conversations.map((c) => c.id);
    expect(ids).toContain('daily');
    expect(ids).not.toContain('c1');
    expect(ids).toHaveLength(20);
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
    expect(buildHistory(result.current.turns)).toEqual([{ role: 'user', content: 'Q' }]);
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
});

describe('useChat scope updates', () => {
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
