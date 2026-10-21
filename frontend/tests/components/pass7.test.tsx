import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import ChatThread from '../../src/components/ChatThread';
import type { ChatTurn, FeedbackPayload } from '../../src/components/ChatMessage';
import ServerNotices from '../../src/components/ServerNotices';
import SummaryDialog from '../../src/components/SummaryDialog';
import TracesPanel from '../../src/components/TracesPanel';
import FeedbackPanel from '../../src/components/FeedbackPanel';
import DocumentCard from '../../src/components/corpus/DocumentCard';
import ServerActivity from '../../src/components/corpus/ServerActivity';
import TraceTab from '../../src/components/inspector/TraceTab';
import RetrievalTab from '../../src/components/inspector/RetrievalTab';
import type { DocCard } from '../../src/components/corpus/types';
import type { TraceDetailResponse } from '../../src/api/types';
import {
  CORPUS_REFRESH_MIN_INTERVAL_MS,
  useCorpusAutoRefresh,
} from '../../src/hooks/useCorpusAutoRefresh';
import { useServerHealth } from '../../src/hooks/useServerHealth';

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

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

const SOURCE = {
  source_number: 1,
  filename: 'a.md',
  page: 1,
  section: 'Intro',
  chunk_id: 'c1',
  text: 'alpha',
};

describe('ChatThread', () => {
  it('offers Ask again on a trailing question with no answer', async () => {
    const onRetry = vi.fn();
    render(
      <ChatThread
        turns={[turn({ id: 'q1', content: 'How long is leave?' })]}
        pending={false}
        engineerMode={false}
        onRetry={onRetry}
      />,
    );
    await userEvent.click(screen.getByTestId('chat-message-ask-again'));
    expect(onRetry).toHaveBeenCalledWith('How long is leave?');
  });

  it('does not offer it while the answer is still coming, or once answered', () => {
    const { rerender } = render(
      <ChatThread turns={[turn({ id: 'q1' })]} pending engineerMode={false} onRetry={vi.fn()} />,
    );
    expect(screen.queryByTestId('chat-message-ask-again')).toBeNull();
    rerender(
      <ChatThread
        turns={[turn({ id: 'q1' }), turn({ id: 'a1', role: 'assistant', content: 'Ten days.' })]}
        pending={false}
        engineerMode={false}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.queryByTestId('chat-message-ask-again')).toBeNull();
  });

  it('flags a weakly supported answer and rates one source as not relevant', async () => {
    const onFeedback = vi.fn<(payload: FeedbackPayload) => void>();
    const turns = [
      turn({ id: 'q1', content: 'Leave?' }),
      turn({
        id: 'a1',
        role: 'assistant',
        content: 'Ten days [1].',
        sources: [SOURCE],
        topRerankScore: 0.2,
        traceId: 'tr',
      }),
    ];
    render(
      <ChatThread
        turns={turns}
        pending={false}
        engineerMode={false}
        feedbackEnabled
        onFeedback={onFeedback}
        weakSupportBelow={0.25}
      />,
    );
    expect(screen.getByTestId('chat-message-weak-support')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('citation-not-relevant'));
    expect(onFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        rating: 'down',
        citationSourceNumber: 1,
        citedFilenames: ['a.md'],
        question: 'Leave?',
      }),
    );
  });

  it('an answer from below-the-bar passages says weak match, not weakly supported', () => {
    const turns = [
      turn({ id: 'q1' }),
      turn({
        id: 'a1',
        role: 'assistant',
        content: 'Foundation models are trained in house [1].',
        sources: [SOURCE],
        topRerankScore: 0.006,
        weakMatch: true,
      }),
    ];
    render(<ChatThread turns={turns} pending={false} engineerMode={false} weakSupportBelow={0.25} />);
    expect(screen.getByTestId('chat-message-weak-match')).toHaveTextContent('weak match');
    expect(screen.queryByTestId('chat-message-weak-support')).toBeNull();
  });

  it('a marked source stays marked and well-supported answers carry no badge', () => {
    const turns = [
      turn({ id: 'q1' }),
      turn({
        id: 'a1',
        role: 'assistant',
        content: 'Ten days [1].',
        sources: [SOURCE],
        topRerankScore: 0.9,
        irrelevantSources: [1],
      }),
    ];
    render(
      <ChatThread
        turns={turns}
        pending={false}
        engineerMode={false}
        feedbackEnabled
        onFeedback={vi.fn()}
        weakSupportBelow={0.25}
      />,
    );
    expect(screen.queryByTestId('chat-message-weak-support')).toBeNull();
    expect(screen.getByTestId('citation-not-relevant')).toBeDisabled();
    expect(screen.getByTestId('citation-not-relevant')).toHaveTextContent('not relevant');
  });

  it('shows a way back to the latest turn once scrolled up', () => {
    const turns = [turn({ id: 'q1' }), turn({ id: 'a1', role: 'assistant', content: 'x' })];
    render(<ChatThread turns={turns} pending={false} engineerMode={false} />);
    const thread = screen.getByTestId('chat-thread');
    Object.defineProperty(thread, 'scrollHeight', { configurable: true, value: 2000 });
    Object.defineProperty(thread, 'clientHeight', { configurable: true, value: 500 });
    thread.scrollTop = 100;
    fireEvent.scroll(thread);
    fireEvent.click(screen.getByTestId('chat-jump-latest'));
    expect(screen.queryByTestId('chat-jump-latest')).toBeNull();
  });
});

describe('useCorpusAutoRefresh', () => {
  it('re-lists when the tab comes back, at most once per interval, and after an outage', () => {
    vi.useFakeTimers();
    const refresh = vi.fn(async () => {});
    let visibility: DocumentVisibilityState = 'visible';
    vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility);
    const { rerender } = renderHook(
      ({ lost }: { lost: boolean }) => useCorpusAutoRefresh(true, lost, refresh),
      { initialProps: { lost: false } },
    );
    const show = () => {
      visibility = 'visible';
      document.dispatchEvent(new Event('visibilitychange'));
    };
    show();
    expect(refresh).not.toHaveBeenCalled(); // boot has just listed
    vi.advanceTimersByTime(CORPUS_REFRESH_MIN_INTERVAL_MS + 1);
    show();
    expect(refresh).toHaveBeenCalledTimes(1);
    show();
    expect(refresh).toHaveBeenCalledTimes(1);

    rerender({ lost: true });
    rerender({ lost: false });
    expect(refresh).toHaveBeenCalledTimes(2);
  });
});

describe('server health', () => {
  it('reports an index built for another setup, and a version change', async () => {
    vi.spyOn(api, 'readiness').mockResolvedValue({
      status: 'ok',
      qdrant: true,
      generation_provider: true,
      llm_provider: 'ollama',
      index_compatible: false,
      index_detail: 'tag differs',
      originals_kept: false,
    });
    vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', version: '0.1.0' });
    const { result } = renderHook(() => useServerHealth(true));
    await waitFor(() => expect(result.current.indexProblem).not.toBeNull());
    expect(result.current.indexProblem).toEqual({ detail: 'tag differs', originalsKept: false });
    expect(result.current.version).toBe('0.1.0');
    expect(result.current.updatedTo).toBeNull();
  });

  it('names only the way out that works with this server', () => {
    const { rerender } = render(
      <ServerNotices
        indexProblem={{ detail: 'tag differs', originalsKept: false }}
        updatedTo={null}
        onReload={vi.fn()}
      />,
    );
    expect(screen.getByTestId('index-mismatch')).toHaveTextContent('tag differs');
    // The server's detail names the way out; a backup can't help without originals.
    expect(screen.getByTestId('index-mismatch')).not.toHaveTextContent('backup');
    rerender(
      <ServerNotices
        indexProblem={{ detail: 'tag differs', originalsKept: true }}
        updatedTo="0.2.0"
        onReload={vi.fn()}
      />,
    );
    expect(screen.getByTestId('index-mismatch')).toHaveTextContent('Restore documents from a backup');
    expect(screen.getByTestId('server-updated')).toHaveTextContent('0.2.0');
  });
});

describe('SummaryDialog', () => {
  it('runs with the focus typed, or a general summary when empty', async () => {
    const onRun = vi.fn();
    const { unmount } = render(
      <SummaryDialog filenames={['a.md']} onRun={onRun} onClose={vi.fn()} />,
    );
    await userEvent.type(screen.getByTestId('summary-focus-input'), 'deadlines{Enter}');
    expect(onRun).toHaveBeenLastCalledWith('deadlines');
    unmount();

    render(<SummaryDialog filenames={['a.md', 'b.md']} onRun={onRun} onClose={vi.fn()} />);
    expect(screen.getByRole('dialog', { name: 'Compare a.md and b.md' })).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('summary-run'));
    expect(onRun).toHaveBeenLastCalledWith(undefined);
  });
});

describe('TracesPanel and FeedbackPanel filters', () => {
  const trace = (id: string, question: string, status: 'ok' | 'error') => ({
    trace_id: id,
    created_at: 1,
    question,
    mode: 'stream' as const,
    status,
    llm_provider: 'ollama',
    total_ms: 1000,
    candidate_count: 1,
    kept_count: 1,
  });

  it('filters traces by question text and by trouble', async () => {
    render(
      <TracesPanel
        traces={[trace('1', 'leave policy', 'ok'), trace('2', 'expense limit', 'error')]}
        error={null}
        onSelect={vi.fn()}
      />,
    );
    await userEvent.type(screen.getByTestId('traces-filter'), 'leave');
    expect(screen.getAllByTestId('traces-panel-row')).toHaveLength(1);
    await userEvent.clear(screen.getByTestId('traces-filter'));
    await userEvent.click(screen.getByTestId('traces-problems-only'));
    expect(screen.getAllByTestId('traces-panel-row')).toHaveLength(1);
    expect(screen.getByTestId('traces-panel-row')).toHaveTextContent('expense limit');
  });

  it('filters feedback by what was rated', async () => {
    const row = (id: string, kind: 'question' | 'summary', citation: number | null) => ({
      id,
      trace_id: null,
      question: `Q${id}`,
      answer_excerpt: '',
      cited_filenames: [],
      rating: 'down' as const,
      citation_source_number: citation,
      created_at: 1,
      kind,
    });
    render(
      <FeedbackPanel
        feedback={[row('1', 'question', null), row('2', 'summary', null), row('3', 'question', 2)]}
        error={null}
        onSelectTrace={vi.fn()}
      />,
    );
    await userEvent.selectOptions(screen.getByTestId('feedback-kind-filter'), 'citation');
    expect(screen.getByText('Q3')).toBeInTheDocument();
    expect(screen.queryByText('Q1')).toBeNull();
    await userEvent.selectOptions(screen.getByTestId('feedback-kind-filter'), 'summary');
    expect(screen.getByText('Q2')).toBeInTheDocument();
    expect(screen.queryByText('Q3')).toBeNull();
  });
});

describe('inspector', () => {
  const detail: TraceDetailResponse = {
    trace_id: 'trace-full-id-123',
    created_at: 1,
    question: 'q',
    mode: 'stream',
    status: 'error',
    config: null,
    candidates: [
      {
        point_id: 'p1',
        filename: 'a.md',
        page: 1,
        section: 'Intro',
        chunk_id: 'c1',
        retrieval_score: 0.5,
        rerank_score: 0.1,
        kept: false,
        drop_reason: 'below_min_score',
        selected_for_context: false,
        neighbor_expanded: false,
      },
    ],
    prompt_messages: null,
    answer: null,
    cited_source_numbers: [],
    timings: {
      embed_ms: 1,
      search_ms: 1,
      rerank_ms: 1,
      generate_ms: 1,
      total_ms: 4,
      condense_ms: 0,
    },
    error: 'The model timed out.',
    query_variants: ['leave days', 'annual leave'],
    request_id: 'req-9',
    usage: { prompt_tokens: 120, completion_tokens: 30 },
  };

  it('shows ids, the error alongside timings, the variants and the token count', () => {
    render(<TraceTab trace={detail} />);
    expect(screen.getByTestId('trace-ids')).toHaveTextContent('trace-full-id-123');
    expect(screen.getByTestId('trace-ids')).toHaveTextContent('req-9');
    expect(screen.getByTestId('trace-error')).toHaveTextContent('The model timed out.');
    expect(screen.getByTestId('trace-variants')).toHaveTextContent('annual leave');
    expect(screen.getByTestId('trace-usage')).toHaveTextContent('120 in');
  });

  it('says why a follow-up was not condensed', () => {
    render(<TraceTab trace={{ ...detail, condense_note: 'standalone' }} />);
    expect(screen.getByTestId('trace-condense-note')).toHaveTextContent('standalone');
    expect(screen.queryByTestId('trace-weak-matches')).toBeNull();
  });

  it('says when the answer came from weak matches', () => {
    render(<TraceTab trace={{ ...detail, weak_matches: true }} />);
    expect(screen.getByTestId('trace-weak-matches')).toHaveTextContent('closest ones');
  });

  it('opens a dropped candidate in the viewer', async () => {
    const onOpenSource = vi.fn();
    render(<RetrievalTab trace={detail} onOpenSource={onOpenSource} />);
    await userEvent.click(screen.getByTestId('retrieval-candidate-open'));
    expect(onOpenSource).toHaveBeenCalledWith('a.md', 'c1');
  });
});

describe('corpus panel additions', () => {
  const card: DocCard = {
    key: 'doc:policy.html',
    filename: 'policy.html',
    status: 'indexed',
    detail: '3 chunks',
    pct: 100,
    tags: [],
    sourceUrl: 'https://example.com/policy.html',
  };

  it('links a URL document to its source, refreshes it, and scopes a chat to it', async () => {
    const onRefresh = vi.fn();
    const onAskAbout = vi.fn();
    render(
      <DocumentCard
        card={card}
        readOnly={false}
        confirming={false}
        deleting={false}
        selectable={false}
        selected={false}
        onToggleSelect={vi.fn()}
        onAskDelete={vi.fn()}
        onCancelDelete={vi.fn()}
        onConfirmDelete={vi.fn()}
        onRefreshFromUrl={onRefresh}
        onAskAbout={onAskAbout}
      />,
    );
    const link = screen.getByTestId('corpus-source-link');
    expect(link).toHaveAttribute('href', 'https://example.com/policy.html');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    expect(link).toHaveTextContent('example.com');
    await userEvent.click(screen.getByTestId('corpus-refresh-url'));
    expect(onRefresh).toHaveBeenCalledWith('policy.html');
    await userEvent.click(screen.getByTestId('corpus-ask-about'));
    expect(onAskAbout).toHaveBeenCalledWith(['policy.html']);
  });

  it('lists jobs this tab is not following and the watched folder, and starts a pass', async () => {
    vi.spyOn(api, 'listActiveJobs').mockResolvedValue({
      jobs: [
        {
          job_id: 'j1',
          filename: 'mine.pdf',
          state: 'embedding',
          chunks_total: 4,
          chunks_done: 1,
          error: null,
          result: null,
        },
        {
          job_id: 'j2',
          filename: 'watched.md',
          state: 'queued',
          chunks_total: 0,
          chunks_done: 0,
          error: null,
          result: null,
        },
      ],
    });
    const status = {
      folder: '/data/inbox',
      last_scan_at: null,
      files: 2,
      problems: [],
      deferred: [],
      retrying: [
        { filename: 'broken.pdf', attempts: 1, next_attempt_at: 0, error: 'Qdrant is down' },
      ],
      gave_up: [],
    };
    vi.spyOn(api, 'watchStatus').mockResolvedValue(status);
    const scan = vi.spyOn(api, 'watchScan').mockResolvedValue(status);
    render(
      <ServerActivity
        workspace=""
        followed={new Set(['mine.pdf'])}
        watchEnabled
        canScan
        onJobsFinished={vi.fn()}
      />,
    );
    expect(await screen.findByTestId('active-jobs')).toHaveTextContent('watched.md');
    expect(screen.getByTestId('active-jobs')).not.toHaveTextContent('mine.pdf');
    expect(await screen.findByTestId('watch-status')).toHaveTextContent('/data/inbox');
    expect(screen.getByTestId('watch-status')).toHaveTextContent('retrying broken.pdf');
    await act(async () => {
      await userEvent.click(screen.getByTestId('watch-scan'));
    });
    expect(scan).toHaveBeenCalled();
  });
});

describe('briefs', () => {
  it('offers a brief for three or more selected documents, up to the server limit', async () => {
    const { default: BulkBar } = await import('../../src/components/corpus/BulkBar');
    const onCompare = vi.fn();
    const props = { reindexable: [], onCompare, onClear: vi.fn(), maxBrief: 3 };
    const { rerender } = render(<BulkBar {...props} selected={['a', 'b', 'c']} />);
    await userEvent.click(screen.getByTestId('corpus-bulk-brief'));
    expect(onCompare).toHaveBeenCalledWith(['a', 'b', 'c']);
    rerender(<BulkBar {...props} selected={['a', 'b', 'c', 'd']} />);
    expect(screen.queryByTestId('corpus-bulk-brief')).toBeNull();
    rerender(<BulkBar {...props} selected={['a', 'b']} />);
    expect(screen.queryByTestId('corpus-bulk-brief')).toBeNull();
    expect(screen.getByTestId('corpus-bulk-compare')).toBeInTheDocument();
  });

  it('keeps a brief task on a saved turn', async () => {
    const { sanitizeTask } = await import('../../src/hooks/chatPersistence');
    expect(sanitizeTask({ mode: 'brief', filenames: ['a', 'b', 'c'] })).toEqual({
      mode: 'brief',
      filenames: ['a', 'b', 'c'],
    });
    expect(sanitizeTask({ mode: 'brief', filenames: ['a'] })).toBeNull();
  });
});
