import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { PublicConfigResponse, TraceSummaryResponse } from '../../src/api/types';
import type { ChatTurn } from '../../src/components/ChatMessage';
import SavedPanel from '../../src/components/SavedPanel';
import SearchPanel from '../../src/components/SearchPanel';
import SettingsModal from '../../src/components/SettingsModal';
import TracesPanel from '../../src/components/TracesPanel';
import DocumentCard from '../../src/components/corpus/DocumentCard';
import FilePreview from '../../src/components/corpus/FilePreview';
import ServerActivity from '../../src/components/corpus/ServerActivity';
import type { DocCard } from '../../src/components/corpus/types';
import { EMPTY_OVERRIDES } from '../../src/utils/overrides';

afterEach(() => {
  vi.restoreAllMocks();
});

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
  document.dispatchEvent(new Event('visibilitychange'));
}

describe('server activity', () => {
  afterEach(() => setVisibility('visible'));

  it('lists recent failures without a watched folder', async () => {
    vi.spyOn(api, 'listActiveJobs').mockResolvedValue({ jobs: [] });
    render(
      <ServerActivity workspace="" followed={new Set()} watchEnabled={false} canScan onJobsFinished={vi.fn()} />,
    );
    expect(await screen.findByText('Recent failures')).toBeInTheDocument();
  });

  it('polls nothing while the tab is hidden, and at once when it is shown', async () => {
    const jobs = vi.spyOn(api, 'listActiveJobs').mockResolvedValue({ jobs: [] });
    const watch = vi.spyOn(api, 'watchStatus').mockResolvedValue({
      folder: '/inbox',
      last_scan_at: null,
      files: 0,
      problems: [],
      deferred: [],
      retrying: [],
      gave_up: [],
    });
    setVisibility('hidden');
    render(
      <ServerActivity workspace="" followed={new Set()} watchEnabled canScan={false} onJobsFinished={vi.fn()} />,
    );
    await act(async () => {});
    expect(jobs).not.toHaveBeenCalled();
    const watchReads = watch.mock.calls.length;

    await act(async () => setVisibility('visible'));
    expect(jobs).toHaveBeenCalledTimes(1);
    expect(watch.mock.calls.length).toBeGreaterThan(watchReads);
  });

  it('retries a failed job the server can run again, and lists finished ones on request', async () => {
    vi.spyOn(api, 'listActiveJobs').mockResolvedValue({ jobs: [] });
    const finished = vi.spyOn(api, 'listFinishedJobs').mockResolvedValue({
      jobs: [
        { job_id: 'u', filename: 'page.html', state: 'failed', error: 'timed out', retryable: true } as never,
        { job_id: 'p', filename: 'upload.pdf', state: 'failed', error: 'bad', retryable: false } as never,
      ],
    });
    const retry = vi
      .spyOn(api, 'retryDocumentJob')
      .mockResolvedValue({ job_id: 'n', filename: 'page.html', state: 'queued' });
    render(
      <ServerActivity workspace="" followed={new Set()} watchEnabled={false} canScan onJobsFinished={vi.fn()} />,
    );
    const details = (await screen.findByText('Recent failures')).closest('details')!;
    await act(async () => {
      details.open = true;
      fireEvent(details, new Event('toggle'));
    });
    await screen.findAllByTestId('finished-job-retry');
    // jsdom fires its own toggle event for the opened details: let that read settle.
    await act(async () => {});
    const buttons = await screen.findAllByTestId('finished-job-retry');
    expect(buttons).toHaveLength(1);
    await userEvent.click(buttons[0]);
    expect(retry).toHaveBeenCalledWith('u', undefined, '');

    await userEvent.click(screen.getByTestId('finished-show-all'));
    expect(finished).toHaveBeenLastCalledWith(false, undefined, '');
    expect(screen.getByText('Recent jobs')).toBeInTheDocument();
  });
});

describe('trace list', () => {
  const trace = (id: string, answerStatus: string): TraceSummaryResponse => ({
    trace_id: id,
    created_at: 1,
    question: `question ${id}`,
    mode: 'stream',
    status: 'ok',
    llm_provider: 'ollama',
    total_ms: 1000,
    candidate_count: 1,
    kept_count: 1,
    answer_status: answerStatus,
    top_rerank_score: 0.42,
  });

  it('asks for traces answered one way, and marks uncited ones', async () => {
    const onAnswerFilter = vi.fn();
    render(
      <TracesPanel
        traces={[trace('a', 'answered'), trace('u', 'uncited')]}
        error={null}
        onSelect={vi.fn()}
        onAnswerFilter={onAnswerFilter}
      />,
    );
    expect(screen.getAllByTestId('traces-panel-status')).toHaveLength(1);
    expect(screen.getAllByText('0.42')).toHaveLength(2);
    await userEvent.selectOptions(screen.getByTestId('traces-answer-filter'), 'unanswered');
    expect(onAnswerFilter).toHaveBeenCalledWith('unanswered');
  });
});

describe('saved answers panel', () => {
  const answer = (id: string, content: string, timestamp: number): ChatTurn => ({
    id,
    role: 'assistant',
    content,
    sources: [],
    timestamp,
    timings: null,
    traceId: null,
    saved: true,
  });

  it('lists, filters, opens and unsaves saved answers', async () => {
    const onOpen = vi.fn();
    const onUnsave = vi.fn();
    render(
      <SavedPanel
        answers={[
          { conversationId: 'c1', conversationTitle: 'Leave', turn: answer('t1', 'Twenty days.', 2), question: 'How long?' },
          { conversationId: 'c2', conversationTitle: 'Travel', turn: answer('t2', 'Economy.', 1), question: 'Which class?' },
        ]}
        onOpen={onOpen}
        onUnsave={onUnsave}
      />,
    );
    expect(screen.getAllByTestId('saved-row')).toHaveLength(2);
    await userEvent.type(screen.getByTestId('saved-filter'), 'economy');
    expect(screen.getAllByTestId('saved-row')).toHaveLength(1);
    await userEvent.click(screen.getByTestId('saved-open'));
    expect(onOpen).toHaveBeenCalledWith('c2', 't2');
    await userEvent.click(screen.getByTestId('saved-unsave'));
    expect(onUnsave).toHaveBeenCalledWith('c2', 't2');
  });

  it('says how to save one when there are none', () => {
    render(<SavedPanel answers={[]} onOpen={vi.fn()} onUnsave={vi.fn()} />);
    expect(screen.getByText(/Nothing saved yet/)).toBeInTheDocument();
  });
});

describe('similar passages', () => {
  it('searches from a result, without a query, and names what it is like', async () => {
    const search = vi.spyOn(api, 'searchPassages').mockResolvedValue({
      results: [
        {
          rank: 1,
          filename: 'a.pdf',
          page: 3,
          section: 'Leave',
          chunk_id: 'c1',
          chunk_ordinal: 1,
          text: 'Annual leave is twenty days.',
          retrieval_score: 0.03,
          rerank_score: 0.9,
        },
      ],
      timings: { embed_ms: 0, search_ms: 0, rerank_ms: 0 },
    } as never);
    render(
      <SearchPanel filenames={[]} tags={[]} resetKey={0} onOpenSource={vi.fn()} onAuthFailure={() => false} />,
    );
    await userEvent.type(screen.getByTestId('search-input'), 'leave');
    await userEvent.click(screen.getByTestId('search-submit'));
    await userEvent.click(await screen.findByTestId('search-similar'));

    expect(search).toHaveBeenLastCalledWith(
      expect.objectContaining({ query: '', like_chunk_id: 'c1' }),
      expect.anything(),
    );
    expect(await screen.findByTestId('search-like')).toHaveTextContent('Passages like a.pdf p.3');
  });
});

describe('fetched document refresh setting', () => {
  const card: DocCard = {
    key: 'doc:page.html',
    filename: 'page.html',
    status: 'indexed',
    detail: '3 chunks',
    pct: 100,
    tags: [],
    sourceUrl: 'https://example.com/page.html',
    urlRefresh: { refresh_minutes: null, last_checked: null, failures: 0, next_due: null },
  };

  it('sets how often the document is checked', async () => {
    const onSetRefreshInterval = vi.fn();
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
        onSetRefreshInterval={onSetRefreshInterval}
      />,
    );
    await userEvent.selectOptions(screen.getByTestId('corpus-refresh-every'), 'never');
    expect(onSetRefreshInterval).toHaveBeenCalledWith('page.html', 0);
    await userEvent.selectOptions(screen.getByTestId('corpus-refresh-every'), 'server default');
    expect(onSetRefreshInterval).toHaveBeenLastCalledWith('page.html', null);
  });
});

describe('file preview', () => {
  it('shows how a file would be indexed, indexing nothing', async () => {
    const preview = vi.spyOn(api, 'previewDocument').mockResolvedValue({
      filename: 'policy.md',
      section_count: 2,
      chunk_count: 3,
      page_count: 1,
      section_labels: ['Leave', 'Travel'],
      first_chunks: [{ ordinal: 1, page: 1, section: 'Leave', text: 'Annual leave is twenty days.' }],
    });
    const upload = vi.spyOn(api, 'uploadDocument');
    render(<FilePreview accept={['.md']} workspace="" />);
    const file = new File(['# Leave'], 'policy.md', { type: 'text/markdown' });
    await userEvent.upload(screen.getByTestId('preview-file-input'), file);

    expect(await screen.findByTestId('preview-result')).toHaveTextContent(
      'policy.md: 3 passages in 2 sections — not indexed',
    );
    expect(screen.getByText('Annual leave is twenty days.')).toBeInTheDocument();
    expect(preview).toHaveBeenCalledWith(file, undefined, '');
    expect(upload).not.toHaveBeenCalled();
  });
});

describe('stricter answers in Settings', () => {
  it('saves the two switches with the other overrides', async () => {
    const onOverridesChange = vi.fn();
    render(
      <SettingsModal
        config={
          {
            llm_provider: 'ollama',
            llm_model: 'llama3.1:8b',
            rerank_top_k: 8,
            max_context_chunks: 6,
            rerank_top_k_limit: 50,
            max_context_chunks_limit: 20,
            llm_temperature: 0,
            llm_temperature_max: 2,
            answer_format: 'default',
          } as PublicConfigResponse
        }
        overrides={EMPTY_OVERRIDES}
        onOverridesChange={onOverridesChange}
        onClose={vi.fn()}
        onSaved={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId('settings-passages-only'));
    await userEvent.click(screen.getByTestId('settings-strong-only'));
    await userEvent.click(screen.getByTestId('settings-save'));
    expect(onOverridesChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ passagesOnly: true, strongMatchesOnly: true }),
    );
  });
});
