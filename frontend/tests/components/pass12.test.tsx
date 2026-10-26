import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { ApiClientError, api } from '../../src/api/client';
import type { PublicConfigResponse, TraceDetailResponse } from '../../src/api/types';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';
import CitationCard from '../../src/components/CitationCard';
import Composer from '../../src/components/Composer';
import ConversationList from '../../src/components/ConversationList';
import DocumentViewer from '../../src/components/DocumentViewer';
import SearchPanel from '../../src/components/SearchPanel';
import ServerNotices from '../../src/components/ServerNotices';
import ServerActivity from '../../src/components/corpus/ServerActivity';
import TraceTab from '../../src/components/inspector/TraceTab';
import { loadMarkdown } from '../../src/components/markdownLoader';

beforeAll(async () => {
  await loadMarkdown();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function makeConfig(overrides: Partial<PublicConfigResponse> = {}): PublicConfigResponse {
  return {
    qdrant_collection: 'docrag_documents',
    dense_embedding_model: 'BAAI/bge-small-en-v1.5',
    sparse_embedding_model: 'Qdrant/BM25',
    reranker_model: 'jinaai/jina-reranker-v2-base-multilingual',
    embedding_model_tag: 'tag',
    llm_provider: 'ollama',
    llm_model: 'llama3.1:8b',
    llm_temperature: 0,
    openai_model: 'gpt-4o-mini',
    openai_available: false,
    ollama_available: true,
    rrf_k: 60,
    dense_retrieval_limit: 50,
    sparse_retrieval_limit: 50,
    fused_top_n: 50,
    rerank_top_k: 8,
    max_context_chunks: 6,
    rerank_min_score: 0.15,
    max_upload_bytes: 52428800,
    rerank_top_k_limit: 50,
    max_context_chunks_limit: 20,
    llm_temperature_max: 2.0,
    feedback_enabled: false,
    ...overrides,
  };
}

function answer(overrides: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id: 'a1',
    role: 'assistant',
    content: 'Twenty days [1].',
    sources: [
      {
        source_number: 1,
        filename: 'leave.pdf',
        page: 2,
        section: 'Leave',
        chunk_id: 'c9',
        text: 'Twenty days',
        rerank_score: 0.82,
      },
    ],
    timestamp: Date.now(),
    timings: null,
    traceId: null,
    ...overrides,
  };
}

describe('citations', () => {
  it('show their score and ask the question again without the passage', async () => {
    const onRegenerate = vi.fn();
    render(
      <ChatMessage
        turn={answer()}
        engineerMode={false}
        regenerateQuestion="How many leave days?"
        regenerateNarrowing={{ page_min: 1, exclude_chunk_ids: ['c1'] }}
        onRetry={vi.fn()}
        onRegenerate={onRegenerate}
      />,
    );
    expect(screen.getByTestId('citation-score')).toHaveTextContent('0.82');
    await userEvent.click(screen.getByRole('button', { name: 'Ask again without source 1' }));
    expect(onRegenerate).toHaveBeenCalledWith('How many leave days?', undefined, {
      page_min: 1,
      exclude_chunk_ids: ['c1', 'c9'],
    });
  });

  it('wait while an answer is being written', async () => {
    const onAskWithout = vi.fn();
    render(
      <CitationCard
        sourceNumber={2}
        filename="a.pdf"
        page={1}
        section="S"
        onAskWithout={onAskWithout}
        askDisabled
      />,
    );
    await userEvent.click(screen.getByTestId('citation-ask-without'));
    expect(onAskWithout).not.toHaveBeenCalled();
    expect(screen.queryByTestId('citation-score')).toBeNull();
  });
});

describe('saved answers', () => {
  it('are toggled from the answer and named in its label', async () => {
    const onToggleSaved = vi.fn();
    const { rerender } = render(
      <ChatMessage turn={answer()} engineerMode={false} onToggleSaved={onToggleSaved} answerNumber={3} />,
    );
    expect(screen.getByRole('article', { name: 'Answer 3' })).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('chat-message-save'));
    expect(onToggleSaved).toHaveBeenCalledWith('a1', true);

    rerender(
      <ChatMessage
        turn={answer({ saved: true })}
        engineerMode={false}
        onToggleSaved={onToggleSaved}
        answerNumber={3}
      />,
    );
    expect(screen.getByRole('article', { name: 'Answer 3, saved' })).toBeInTheDocument();
    expect(screen.getByTestId('chat-message-save')).toHaveAttribute('aria-pressed', 'true');
  });

  it('are a filter of the conversation list, which reports its order', async () => {
    const onSwitch = vi.fn();
    const onVisibleOrder = vi.fn();
    render(
      <ConversationList
        conversations={[
          { id: 'a', title: 'Plain', updatedAt: Date.now(), turnCount: 1 },
          {
            id: 'b',
            title: 'With a saved answer',
            updatedAt: Date.now() - 1000,
            turnCount: 2,
            savedCount: 1,
            firstSavedTurnId: 't7',
          },
        ]}
        activeId="a"
        onSwitch={onSwitch}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        onVisibleOrder={onVisibleOrder}
      />,
    );
    expect(onVisibleOrder).toHaveBeenLastCalledWith(['a', 'b']);
    expect(screen.getAllByTestId('conversation-group')[0]).toHaveTextContent('Today');
    await userEvent.click(screen.getByTestId('conversation-filter-saved'));
    expect(onVisibleOrder).toHaveBeenLastCalledWith(['b']);
    await userEvent.click(screen.getByText('With a saved answer'));
    expect(onSwitch).toHaveBeenCalledWith('b', 't7');
  });
});

describe('server notices', () => {
  it('say which service the server can no longer reach', () => {
    render(
      <ServerNotices
        indexProblem={null}
        servicesDown={['qdrant', 'llm']}
        llmProvider="ollama"
        updatedTo={null}
        onReload={vi.fn()}
      />,
    );
    expect(screen.getByTestId('qdrant-down')).toHaveTextContent('vector store');
    expect(screen.getByTestId('llm-down')).toHaveTextContent('(ollama)');
  });
});

describe('server activity', () => {
  it('shows skipped files, cancels a job from elsewhere and lists recent failures', async () => {
    vi.spyOn(api, 'listActiveJobs').mockResolvedValue({
      jobs: [
        {
          job_id: 'j1',
          filename: 'watched.md',
          state: 'embedding',
          chunks_total: 4,
          chunks_done: 1,
          error: null,
          result: null,
        } as never,
      ],
    });
    vi.spyOn(api, 'watchStatus').mockResolvedValue({
      folder: '/inbox',
      last_scan_at: null,
      files: 3,
      problems: ['/inbox/link.pdf: not a regular file'],
      deferred: ['big.pdf'],
      retrying: [],
      gave_up: [],
    });
    const cancel = vi
      .spyOn(api, 'cancelDocumentJob')
      .mockRejectedValue(new ApiClientError('It has started writing; it has to finish.', 409));
    vi.spyOn(api, 'listFinishedJobs').mockResolvedValue({
      jobs: [{ job_id: 'j0', filename: 'broken.pdf', state: 'failed', error: 'Unreadable PDF.' } as never],
    });
    render(
      <ServerActivity workspace="" followed={new Set()} watchEnabled canScan onJobsFinished={vi.fn()} />,
    );
    expect(await screen.findByTestId('watch-problem')).toHaveTextContent('link.pdf');
    expect(screen.getByTestId('watch-deferred')).toHaveTextContent('big.pdf');
    await userEvent.click(await screen.findByRole('button', { name: 'Cancel indexing watched.md' }));
    expect(cancel).toHaveBeenCalledWith('j1', undefined, '');
    expect(await screen.findByTestId('active-job-cancel-error')).toHaveTextContent('has to finish');
    await act(async () => {
      fireEvent(screen.getByText('Recent failures').closest('details')!, new Event('toggle'));
    });
    expect(await screen.findByTestId('failed-job')).toHaveTextContent('broken.pdf: Unreadable PDF.');
  });
});

describe('the composer scope', () => {
  it('takes an upload date and reads a page range typed back to front', async () => {
    const onUploadedAfterChange = vi.fn();
    render(
      <Composer
        onSubmit={vi.fn()}
        disabled={false}
        pending={false}
        onCancel={vi.fn()}
        scopeLabel="a.txt"
        indexedFilenames={['a.txt']}
        selectedFilenames={['a.txt']}
        onSelectedFilenamesChange={vi.fn()}
        config={makeConfig()}
        provider="ollama"
        onProviderChange={vi.fn()}
        providerLabel="llama3.1:8b"
        pageRange={{ min: 9, max: 2 }}
        onPageRangeChange={vi.fn()}
        onUploadedAfterChange={onUploadedAfterChange}
      />,
    );
    await userEvent.click(screen.getByTestId('composer-scope-button'));
    expect(screen.getByTestId('scope-pages-order')).toHaveTextContent('pages 2–9');
    fireEvent.change(screen.getByTestId('scope-uploaded-after'), { target: { value: '2026-01-15' } });
    expect(onUploadedAfterChange).toHaveBeenCalledWith(new Date(2026, 0, 15).getTime() / 1000);
  });
});

describe('search', () => {
  it('sends its filters and marks the matching sentence', async () => {
    const search = vi.spyOn(api, 'searchPassages').mockResolvedValue({
      timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1 },
      results: [
        {
          rank: 1,
          filename: 'office.txt',
          page: 1,
          section: 'Intro',
          chunk_id: 'c1',
          chunk_ordinal: 1,
          text: 'The office opens at nine. Parking passes are issued by reception.',
          retrieval_score: 0.03,
          rerank_score: 0.9,
          spans: [{ start: 26, end: 66, quote: 'Parking passes are issued by reception.' }],
        },
      ],
    });
    const onOpenSource = vi.fn();
    render(
      <SearchPanel
        filenames={['office.txt']}
        tags={[]}
        narrowing={{ page_min: 1, page_max: 4 }}
        resetKey={0}
        onOpenSource={onOpenSource}
        onAuthFailure={() => false}
      />,
    );
    expect(screen.getByTestId('search-scope')).toHaveTextContent('p.1–4');
    await userEvent.click(screen.getByText('Filters'));
    await userEvent.type(screen.getByTestId('search-filter-section'), 'Intro');
    await userEvent.type(screen.getByTestId('search-filter-count'), '3');
    await userEvent.type(screen.getByTestId('search-input'), 'parking');
    await userEvent.click(screen.getByTestId('search-submit'));

    expect(await screen.findByTestId('search-match')).toHaveTextContent(
      'Parking passes are issued by reception.',
    );
    expect(search).toHaveBeenCalledWith(
      {
        query: 'parking',
        filenames: ['office.txt'],
        page_min: 1,
        page_max: 4,
        section: 'Intro',
        rerank_top_k: 3,
      },
      expect.anything(),
    );
    await userEvent.click(screen.getByTestId('search-result'));
    expect(onOpenSource).toHaveBeenCalledWith('office.txt', 'c1', [
      'Parking passes are issued by reception.',
    ]);
  });
});

describe('the document viewer', () => {
  it('asks a question about one passage', async () => {
    vi.spyOn(api, 'getDocumentContent').mockResolvedValue({
      filename: 'guide.md',
      chunks: [{ chunk_id: 'c1', page: 1, section: 'Intro', text: 'First chunk.', chunk_ordinal: 1 }],
      total_chunks: 1,
    });
    const onAsk = vi.fn();
    render(
      <DocumentViewer filename="guide.md" chunkId={null} onClose={vi.fn()} onAskAboutPassage={onAsk} />,
    );
    await userEvent.click(await screen.findByTestId('viewer-ask'));
    await userEvent.type(screen.getByTestId('viewer-ask-input'), 'What comes first?');
    await userEvent.click(screen.getByTestId('viewer-ask-submit'));
    expect(onAsk).toHaveBeenCalledWith('What comes first?', 'c1');
    await waitFor(() => expect(screen.queryByTestId('viewer-ask-input')).toBeNull());
  });
});

describe('the trace tab', () => {
  it('shows the answer status and history left out of the prompt', () => {
    const trace: TraceDetailResponse = {
      trace_id: 't1',
      created_at: 1,
      question: 'q',
      mode: 'sync',
      status: 'ok',
      config: null,
      candidates: [],
      prompt_messages: null,
      answer: 'a',
      cited_source_numbers: [],
      timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4, condense_ms: 0 },
      error: null,
      answer_status: 'uncited',
      truncated: true,
      dropped_history_messages: 2,
    };
    render(<TraceTab trace={trace} />);
    expect(screen.getByTestId('trace-answer-status')).toHaveTextContent(
      'answer uncited · cut off at the token limit',
    );
    expect(screen.getByTestId('trace-dropped-history')).toHaveTextContent('2 of the earliest');
  });
});
