import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useChat } from '../../src/hooks/useChat';
import { isEmptyScope, sanitizeTurn } from '../../src/hooks/chatPersistence';
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
  localStorage.clear();
  setWorkspace('');
});

const turn = (id: string, role: ChatTurn['role'], content: string, extra?: Partial<ChatTurn>): ChatTurn => ({
  id,
  role,
  content,
  sources: [],
  timestamp: 1,
  timings: null,
  traceId: null,
  ...extra,
});

describe('pass 14 conversation storage', () => {
  it('an "export all" file keeps each conversation\'s dates through an import', () => {
    const { result } = renderHook(() => useChat());
    const old = Date.now() - 30 * 86_400_000;
    act(() => {
      result.current.importConversations([
        { raw: { turns: [turn('q', 'user', 'Old?'), turn('a', 'assistant', 'Old.')], createdAt: old, updatedAt: old } },
      ]);
    });

    const exported = JSON.parse(result.current.allConversationsJson());
    const [item] = exported.docrag_conversations;
    expect(item.updatedAt).toBe(old);
    localStorage.clear();
    const fresh = renderHook(() => useChat()).result;
    act(() => {
      fresh.current.importConversations(
        exported.docrag_conversations.map((raw: { title: string }) => ({ raw, title: raw.title })),
      );
    });
    const imported = fresh.current.conversations.find((c) => c.title === item.title);
    // Still a month old, not "used just now".
    expect(imported?.updatedAt).toBe(old);
  });

  it('making room for an import never removes a conversation holding a saved answer', () => {
    const { result } = renderHook(() => useChat());
    const now = Date.now();
    act(() => {
      for (let n = 0; n < 99; n += 1) {
        result.current.importConversations([
          {
            raw: {
              turns: [
                turn(`q${n}`, 'user', `Q ${n}?`),
                turn(`a${n}`, 'assistant', `A ${n}.`, n === 0 ? { saved: true } : {}),
              ],
              createdAt: now - 1_000_000 + n,
              updatedAt: now - 1_000_000 + n,
            },
          },
        ]);
      }
    });
    const saved = result.current.conversations.find((c) =>
      c.savedCount ? c.savedCount > 0 : false,
    );
    expect(saved).toBeDefined();

    const items = [1, 2, 3].map((n) => ({
      raw: { turns: [turn(`x${n}`, 'user', `New ${n}?`), turn(`y${n}`, 'assistant', 'New.')] },
    }));
    act(() => {
      result.current.importConversations(items, { makeRoom: true });
    });

    expect(result.current.conversations.some((c) => c.id === saved!.id)).toBe(true);
  });

  it('a new chat over an empty one clears its dates and pages too', () => {
    const { result } = renderHook(() => useChat());
    act(() => {
      result.current.setScope({ filenames: [], tags: [], uploadedBefore: 1_700_000_000 });
    });
    expect(isEmptyScope(result.current.scope)).toBe(false);
    act(() => result.current.newConversation());
    expect(isEmptyScope(result.current.scope)).toBe(true);
  });
});

describe('pass 14 asking', () => {
  it('a busy retry asks in the workspace the question was asked in', async () => {
    vi.useFakeTimers();
    try {
      setWorkspace('legal');
      const workspaces: (string | undefined)[] = [];
      askQuestionStreamMock
        .mockImplementationOnce(async (_q, _p, _o, _f, _h, _handlers, _signal, options) => {
          workspaces.push(options?.workspace);
          throw new ApiClientError('Busy', 429, 2);
        })
        .mockImplementationOnce(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers, _s, options) => {
          workspaces.push(options?.workspace);
          handlers.onDone?.('Ten days [1].', [], ZERO_TIMINGS, 't1', false, {});
        });
      const { result } = renderHook(() => useChat());
      let asked!: Promise<void>;
      act(() => {
        asked = result.current.ask('Leave?');
      });
      await act(async () => {
        await Promise.resolve();
      });
      // The reader moves to another workspace during the countdown.
      setWorkspace('archive');
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
        await asked;
      });
      expect(workspaces).toEqual(['legal', 'legal']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('picked passages left out are kept on the answer, and survive a reload', async () => {
    askQuestionStreamMock.mockImplementationOnce(
      async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
        handlers.onDone?.('Alpha [1].', [], ZERO_TIMINGS, 't1', false, { passagesLeftOut: 3 });
      },
    );
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('Alpha?');
    });
    const answer = result.current.turns.find((t) => t.role === 'assistant')!;
    expect(answer.passagesLeftOut).toBe(3);
    expect(sanitizeTurn(answer)?.passagesLeftOut).toBe(3);
    expect(sanitizeTurn({ ...answer, passagesLeftOut: -1 } as ChatTurn)?.passagesLeftOut).toBeUndefined();
  });
});
