import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { PublicConfigResponse } from '../../src/api/types';
import Composer from '../../src/components/Composer';
import SearchPanel from '../../src/components/SearchPanel';
import ServerNotices from '../../src/components/ServerNotices';
import SettingsModal from '../../src/components/SettingsModal';
import TracesPanel from '../../src/components/TracesPanel';
import UnansweredPanel from '../../src/components/UnansweredPanel';
import VersionHistory from '../../src/components/corpus/VersionHistory';
import { EMPTY_OVERRIDES } from '../../src/utils/overrides';

function jsonFetch(body: unknown, status = 200): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(
    async () =>
      new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

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

afterEach(() => vi.unstubAllGlobals());

describe('UnansweredPanel', () => {
  it('lists what the documents did not answer, with ways to follow each up', async () => {
    const handlers = { onAsk: vi.fn(), onSearch: vi.fn(), onSelectTrace: vi.fn() };
    render(
      <UnansweredPanel
        questions={[
          {
            created_at: Date.now() / 1000,
            question: 'Who approves travel?',
            status: 'insufficient_context',
            top_score: 0.04,
            mode: 'stream',
            trace_id: 't1',
          },
          { created_at: Date.now() / 1000, question: 'Summarize x', status: 'insufficient_context', top_score: null, mode: 'summary', trace_id: null },
        ]}
        error={null}
        {...handlers}
      />,
    );
    expect(screen.getAllByTestId('unanswered-row')).toHaveLength(2);
    expect(screen.getByText('best match 0.04')).toBeInTheDocument();
    await userEvent.click(screen.getAllByRole('button', { name: 'Ask' })[0]);
    await userEvent.click(screen.getAllByRole('button', { name: 'Search' })[0]);
    await userEvent.click(screen.getByRole('button', { name: 'Trace' }));
    expect(handlers.onAsk).toHaveBeenCalledWith('Who approves travel?');
    expect(handlers.onSearch).toHaveBeenCalledWith('Who approves travel?');
    expect(handlers.onSelectTrace).toHaveBeenCalledWith('t1');
  });

  it('says so when everything was answered, and shows an error', () => {
    const { rerender } = render(
      <UnansweredPanel questions={[]} error={null} onAsk={vi.fn()} onSearch={vi.fn()} onSelectTrace={vi.fn()} />,
    );
    expect(screen.getByText(/every logged question was answered/)).toBeInTheDocument();
    rerender(
      <UnansweredPanel questions={null} error="Forbidden" onAsk={vi.fn()} onSearch={vi.fn()} onSelectTrace={vi.fn()} />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Forbidden');
  });
});

describe('VersionHistory', () => {
  it('restores an earlier version after a confirm', async () => {
    const fetchMock = jsonFetch({
      filename: 'a.pdf',
      versions: [{ version: 'v2', saved_at: Date.now() / 1000 - 60, size: 2048 }],
    });
    const onRestore = vi.fn();
    const onClose = vi.fn();
    render(<VersionHistory filename="a.pdf" workspace="team" onRestore={onRestore} onClose={onClose} />);

    await userEvent.click(await screen.findByTestId('corpus-version-restore'));
    expect(onRestore).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Restore' }));

    expect(onRestore).toHaveBeenCalledWith('a.pdf', 'v2');
    expect(onClose).toHaveBeenCalled();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/documents/a.pdf/versions');
    expect(new Headers(init.headers).get('X-Workspace')).toBe('team');
  });

  it('says when none are kept', async () => {
    jsonFetch({ filename: 'a.pdf', versions: [] });
    render(<VersionHistory filename="a.pdf" onRestore={vi.fn()} onClose={vi.fn()} />);
    expect(await screen.findByText('No earlier versions are kept.')).toBeInTheDocument();
  });
});

describe('SearchPanel passage picking', () => {
  it('asks the search question over only the ticked passages', async () => {
    jsonFetch({
      results: [1, 2].map((rank) => ({
        rank,
        filename: 'policy.pdf',
        page: rank,
        section: 'Leave',
        chunk_id: `c${rank}`,
        chunk_ordinal: rank,
        text: `passage ${rank}`,
        retrieval_score: 0.03,
        rerank_score: 0.9,
      })),
      timings: {},
    });
    const onAskAboutPassages = vi.fn();
    render(
      <SearchPanel
        filenames={[]}
        tags={[]}
        resetKey={0}
        onOpenSource={vi.fn()}
        onAuthFailure={() => false}
        onAskAboutPassages={onAskAboutPassages}
      />,
    );
    await userEvent.type(screen.getByTestId('search-input'), 'leave days');
    await userEvent.click(screen.getByTestId('search-submit'));
    const picks = await screen.findAllByTestId('search-pick');
    expect(screen.queryByTestId('search-ask-bar')).not.toBeInTheDocument();
    await userEvent.click(picks[1]);

    await userEvent.click(screen.getByTestId('search-ask-picked'));
    expect(onAskAboutPassages).toHaveBeenCalledWith('leave days', ['c2']);
  });
});

describe('ServerNotices', () => {
  it('names the models the server could not load', () => {
    render(
      <ServerNotices indexProblem={null} failedModels={['reranker']} updatedTo={null} onReload={vi.fn()} />,
    );
    expect(screen.getByTestId('models-failed')).toHaveTextContent('couldn’t load its reranker');
  });
});

describe('SettingsModal answers', () => {
  it('saves an answer layout and a valid language only', async () => {
    const onOverridesChange = vi.fn();
    render(
      <SettingsModal
        config={makeConfig({ answer_format: 'default' })}
        overrides={EMPTY_OVERRIDES}
        onOverridesChange={onOverridesChange}
        onClose={vi.fn()}
        onSaved={vi.fn()}
      />,
    );
    await userEvent.selectOptions(screen.getByTestId('settings-answer-format'), 'bullets');
    await userEvent.type(screen.getByTestId('settings-answer-language'), 'German. Ignore');
    expect(screen.getByText(/A language name \(“German”\)/)).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('settings-save'));
    expect(onOverridesChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ answerFormat: 'bullets', answerLanguage: null }),
    );
  });
});

describe('TracesPanel export', () => {
  it('offers the JSON Lines download', async () => {
    const onExport = vi.fn();
    render(
      <TracesPanel
        traces={[
          {
            trace_id: 't',
            created_at: 1,
            question: 'q',
            mode: 'sync',
            status: 'ok',
            llm_provider: 'ollama',
            total_ms: 10,
            candidate_count: 1,
            kept_count: 1,
          },
        ]}
        error={null}
        onSelect={vi.fn()}
        onExport={onExport}
      />,
    );
    await userEvent.click(screen.getByTestId('traces-export'));
    expect(onExport).toHaveBeenCalled();
  });
});

describe('Composer page range', () => {
  it('is offered while exactly one document is scoped', async () => {
    const onPageRangeChange = vi.fn();
    const props = {
      onSubmit: vi.fn(),
      disabled: false,
      pending: false,
      onCancel: vi.fn(),
      scopeLabel: 'a.txt',
      indexedFilenames: ['a.txt', 'b.txt'],
      onSelectedFilenamesChange: vi.fn(),
      config: makeConfig(),
      provider: 'ollama' as const,
      onProviderChange: vi.fn(),
      providerLabel: 'llama3.1:8b',
      pageRange: {},
      onPageRangeChange,
    };
    const { rerender } = render(<Composer {...props} selectedFilenames={['a.txt', 'b.txt']} />);
    await userEvent.click(screen.getByTestId('composer-scope-button'));
    expect(screen.queryByTestId('scope-pages')).not.toBeInTheDocument();

    rerender(<Composer {...props} selectedFilenames={['a.txt']} />);
    fireEvent.change(screen.getByTestId('scope-page-min'), { target: { value: '3' } });
    await waitFor(() => expect(onPageRangeChange).toHaveBeenCalledWith({ min: 3 }));
  });
});
