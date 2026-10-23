import { act, render, renderHook, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { QuestionStreamHandlers } from '../../src/api/client';
import ChatThread from '../../src/components/ChatThread';
import type { ChatTurn } from '../../src/components/ChatMessage';
import SearchPanel from '../../src/components/SearchPanel';
import StreamingSkeleton from '../../src/components/StreamingSkeleton';
import { sanitizeTurn } from '../../src/hooks/chatPersistence';
import { useChat } from '../../src/hooks/useChat';
import type { TimingsResponse } from '../../src/api/types';
import { formatElapsed } from '../../src/utils/elapsed';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return {
    ...actual,
    api: { ...actual.api, askQuestionStream: vi.fn(), searchPassages: vi.fn() },
  };
});

const askMock = vi.mocked(api.askQuestionStream);
const searchMock = vi.mocked(api.searchPassages);

const ZERO_TIMINGS: TimingsResponse = {
  embed_ms: 0,
  search_ms: 0,
  rerank_ms: 0,
  generate_ms: 0,
  total_ms: 0,
  condense_ms: 0,
};

const SOURCE = {
  source_number: 1,
  filename: 'a.md',
  page: 1,
  section: 'Intro',
  chunk_id: 'c1',
  text: 'alpha',
};

function turn(overrides: Partial<ChatTurn>): ChatTurn {
  return {
    id: 't',
    role: 'user',
    content: 'q',
    sources: [],
    timestamp: 0,
    timings: null,
    traceId: null,
    ...overrides,
  };
}

afterEach(() => {
  askMock.mockReset();
  searchMock.mockReset();
  localStorage.clear();
  vi.useRealTimers();
});

describe('waiting for a local model', () => {
  it('shows how long the question has waited, and the reading estimate', () => {
    vi.useFakeTimers();
    const now = Date.now();
    render(<StreamingSkeleton stage="reading the passages…" startedAt={now - 42_000} etaMs={65_000} />);
    expect(screen.getByTestId('streaming-elapsed')).toHaveTextContent('0:42 · reading takes about 1:05');
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(screen.getByTestId('streaming-elapsed')).toHaveTextContent('0:45');
    expect(formatElapsed(125_000)).toBe('2:05');
  });

  it('tracks the stage timing and the reading estimate through a stream', async () => {
    let release!: () => void;
    askMock.mockImplementation((_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onStage?.('searching');
      handlers.onSources?.([SOURCE], 't');
      handlers.onStage?.('reading', undefined, { promptTokens: 4000, etaMs: 50_000 });
      return new Promise<void>((resolve) => {
        release = () => {
          handlers.onDelta?.('Alpha [1].');
          handlers.onDone?.('Alpha [1].', [SOURCE], ZERO_TIMINGS, 't', false, { answerStatus: 'answered' });
          resolve();
        };
      });
    });
    const { result } = renderHook(() => useChat());
    let asked!: Promise<void>;
    act(() => {
      asked = result.current.ask('alpha?');
    });
    expect(result.current.stage).toBe('reading the passages…');
    expect(result.current.stageTiming?.etaMs).toBe(50_000);
    expect(result.current.stageTiming?.askedAt).toBeLessThanOrEqual(Date.now());
    await act(async () => {
      release();
      await asked;
    });
    expect(result.current.stageTiming).toBeNull();
  });

  it('says what happens after the text has started (checking citations)', () => {
    const turns = [turn({ id: 'q1' }), turn({ id: 'a1', role: 'assistant', content: 'Alpha.' })];
    const { rerender } = render(
      <ChatThread turns={turns} pending stage="checking citations…" engineerMode={false} />,
    );
    expect(screen.getByTestId('chat-message-stage')).toHaveTextContent('checking citations…');
    expect(screen.getByTestId('chat-status')).toHaveTextContent('checking citations…');
    rerender(<ChatThread turns={turns} pending stage="generating answer…" engineerMode={false} />);
    expect(screen.queryByTestId('chat-message-stage')).toBeNull();
    expect(screen.getByTestId('chat-status')).toHaveTextContent('Answer streaming.');
  });
});

describe('a "not found" answer', () => {
  it('keeps the passages it read and records the scope it was asked in', async () => {
    askMock.mockImplementation(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onSources?.([SOURCE], 't');
      handlers.onDone?.(
        'The provided documents do not contain enough information.',
        [],
        ZERO_TIMINGS,
        't',
        false,
        { answerStatus: 'insufficient_context' },
      );
    });
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('gym?', undefined, undefined, ['a.md']);
    });
    const answer = result.current.turns[1];
    expect(answer.notFound).toBe(true);
    expect(answer.scoped).toBe(true);
    expect(answer.contextSources).toEqual([{ ...SOURCE, text: '' }]);
    const saved = sanitizeTurn(JSON.parse(JSON.stringify(answer)));
    expect(saved?.contextSources).toHaveLength(1);
    expect(saved?.scoped).toBe(true);
  });

  it('explains itself and offers to ask everywhere or search the passages', async () => {
    const onAskEverywhere = vi.fn();
    const onSearchPassages = vi.fn();
    const onOpenSource = vi.fn();
    const turns = [
      turn({ id: 'q1', content: 'Can I expense a gym?' }),
      turn({
        id: 'a1',
        role: 'assistant',
        content: 'The provided documents do not contain enough information.',
        notFound: true,
        scoped: true,
        contextSources: [{ ...SOURCE, text: '' }],
      }),
    ];
    render(
      <ChatThread
        turns={turns}
        pending={false}
        engineerMode={false}
        onRetry={vi.fn()}
        onAskEverywhere={onAskEverywhere}
        onSearchPassages={onSearchPassages}
        onOpenSource={onOpenSource}
      />,
    );
    expect(screen.getByTestId('chat-message-not-found')).toHaveTextContent('not in the selected documents');
    expect(screen.getByTestId('chat-message-notes')).toHaveTextContent(
      'Nothing in the selected documents answers this.',
    );
    await userEvent.click(screen.getByTestId('chat-message-ask-everywhere'));
    expect(onAskEverywhere).toHaveBeenCalledWith('Can I expense a gym?');
    await userEvent.click(screen.getByTestId('chat-message-search-passages'));
    expect(onSearchPassages).toHaveBeenCalledWith('Can I expense a gym?');
    expect(screen.getByTestId('chat-message-closest')).toHaveTextContent('Passages it read (1)');
  });

  it('asked over everything, it offers no "ask across all documents"', () => {
    const turns = [
      turn({ id: 'q1', content: 'Gym?' }),
      turn({ id: 'a1', role: 'assistant', content: 'Not covered.', notFound: true }),
    ];
    render(
      <ChatThread
        turns={turns}
        pending={false}
        engineerMode={false}
        onRetry={vi.fn()}
        onAskEverywhere={vi.fn()}
        onSearchPassages={vi.fn()}
      />,
    );
    expect(screen.getByTestId('chat-message-not-found')).toHaveTextContent('not in your documents');
    expect(screen.queryByTestId('chat-message-ask-everywhere')).toBeNull();
    expect(screen.getByTestId('chat-message-search-passages')).toBeInTheDocument();
  });
});

describe('auto-cited answers', () => {
  it('are flagged from the done frame, kept on reload, and explained', async () => {
    askMock.mockImplementation(async (_q, _p, _o, _f, _h, handlers: QuestionStreamHandlers) => {
      handlers.onDone?.('Alpha [1].', [SOURCE], ZERO_TIMINGS, 't', false, {
        answerStatus: 'answered',
        autoCited: true,
      });
    });
    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.ask('alpha?');
    });
    const answer = result.current.turns[1];
    expect(answer.autoCited).toBe(true);
    expect(sanitizeTurn(JSON.parse(JSON.stringify(answer)))?.autoCited).toBe(true);
    render(<ChatThread turns={result.current.turns} pending={false} engineerMode={false} />);
    expect(screen.getByTestId('chat-message-auto-cited')).toBeInTheDocument();
    expect(screen.getByTestId('chat-message-notes')).toHaveTextContent('added by matching');
  });
});

describe('SearchPanel', () => {
  it('runs a search asked for from elsewhere', async () => {
    searchMock.mockResolvedValue({ results: [], timings: { embed_ms: 0, search_ms: 0, rerank_ms: 0 } });
    const props = { filenames: [], tags: [], resetKey: 0, onOpenSource: vi.fn(), onAuthFailure: () => false };
    const { rerender } = render(<SearchPanel {...props} />);
    rerender(<SearchPanel {...props} request={{ query: 'gym membership', id: 1 }} />);
    expect(screen.getByTestId('search-input')).toHaveValue('gym membership');
    await vi.waitFor(() => expect(searchMock).toHaveBeenCalledTimes(1));
    expect(searchMock.mock.calls[0][0]).toEqual({ query: 'gym membership' });
  });
});

describe('the inspector on a "not found" answer', () => {
  it('says why there are no sources', async () => {
    const { notFoundSourcesMessage } = await import('../../src/appHelpers');
    expect(notFoundSourcesMessage(turn({ role: 'assistant' }))).toBeUndefined();
    expect(
      notFoundSourcesMessage(
        turn({ role: 'assistant', notFound: true, contextSources: [SOURCE, { ...SOURCE, source_number: 2 }] }),
      ),
    ).toMatch(/read 2 passages and none answered/);
    expect(notFoundSourcesMessage(turn({ role: 'assistant', notFound: true, scoped: true }))).toMatch(
      /selected documents/,
    );
  });
});

describe('pass 10: asking about a selection', () => {
  it('limits the scope to the documents a question can carry, each once', async () => {
    const { scopeForAskAbout } = await import('../../src/appHelpers');
    const many = Array.from({ length: 503 }, (_, n) => `doc-${n}.md`);
    const result = scopeForAskAbout([...many, 'doc-0.md']);
    expect(result.selected).toBe(503);
    expect(result.filenames).toHaveLength(500);
    expect(new Set(result.filenames).size).toBe(500);
    expect(scopeForAskAbout(['a.md', 'a.md'])).toEqual({ filenames: ['a.md'], selected: 1 });
  });
});

describe('pass 10: server activity', () => {
  it('re-lists the documents when a job ends even if another one started meanwhile', async () => {
    vi.useFakeTimers();
    try {
      const { default: ServerActivity, ACTIVE_JOBS_POLL_MS } = await import(
        '../../src/components/corpus/ServerActivity'
      );
      const job = (id: string) => ({
        job_id: id,
        filename: `${id}.md`,
        state: 'embedding' as const,
        chunks_total: 0,
        chunks_done: 0,
        error: null,
        result: null,
      });
      const listed = vi
        .spyOn(api, 'listActiveJobs')
        .mockResolvedValueOnce({ jobs: [job('a')] })
        .mockResolvedValueOnce({ jobs: [job('b')] });
      const finished = vi.fn();
      render(
        <ServerActivity
          workspace=""
          followed={new Set()}
          watchEnabled={false}
          canScan={false}
          onJobsFinished={finished}
        />,
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(finished).not.toHaveBeenCalled();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(ACTIVE_JOBS_POLL_MS);
      });
      expect(listed).toHaveBeenCalledTimes(2);
      expect(finished).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('pass 10: copy is announced', () => {
  it('says on the thread status line that the answer was copied', async () => {
    const writeText = vi.fn(async () => undefined);
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    try {
      render(
        <ChatThread
          turns={[
            { id: 'q', role: 'user', content: 'Q?', sources: [], timestamp: 1, timings: null, traceId: null },
            { id: 'a', role: 'assistant', content: 'Answer.', sources: [], timestamp: 2, timings: null, traceId: null },
          ]}
          pending={false}
          engineerMode={false}
        />,
      );
      await act(async () => {
        await userEvent.click(screen.getByTestId('chat-message-copy'));
      });
      expect(writeText).toHaveBeenCalled();
      expect(screen.getByTestId('chat-status')).toHaveTextContent('Answer copied.');
      expect(document.querySelectorAll('[aria-live], [role="status"]')).toHaveLength(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('pass 10: reading a conversation file', () => {
  it('tells an export of all conversations from one, and anything else', async () => {
    const { parseConversationFile } = await import('../../src/appHelpers');
    expect(parseConversationFile('not json')).toEqual({ kind: 'none' });
    expect(parseConversationFile('null')).toEqual({ kind: 'none' });
    expect(parseConversationFile('[{"role": "user"}]')).toEqual({
      kind: 'one',
      raw: [{ role: 'user' }],
    });
    expect(
      parseConversationFile('{"docrag_conversations": [{"title": "A", "turns": []}, 3]}'),
    ).toEqual({
      kind: 'many',
      items: [
        { raw: { title: 'A', turns: [] }, title: 'A' },
        { raw: 3, title: undefined },
      ],
    });
  });
});
