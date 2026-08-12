import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError } from '../../src/api/client';
import Inspector from '../../src/components/Inspector';
import { clearTraceCache } from '../../src/hooks/useTrace';
import type { CitationResponse, TraceCandidateResponse, TraceDetailResponse } from '../../src/api/types';

const getTrace = vi.fn();
vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>(
    '../../src/api/client',
  );
  return { ...actual, api: { getTrace: (...args: unknown[]) => getTrace(...args) } };
});

function makeSource(overrides: Partial<CitationResponse> = {}): CitationResponse {
  return {
    source_number: 1,
    filename: 'guide.md',
    page: 3,
    section: 'Setup',
    chunk_id: 'chunk-a',
    text: 'Run docker compose up to start Qdrant and the API.',
    ...overrides,
  };
}

function makeCandidate(overrides: Partial<TraceCandidateResponse> = {}): TraceCandidateResponse {
  return {
    point_id: 'p1',
    filename: 'guide.md',
    page: 3,
    section: 'Setup',
    chunk_id: 'chunk-a',
    retrieval_score: 0.5,
    rerank_score: 0.91,
    kept: true,
    drop_reason: null,
    selected_for_context: true,
    neighbor_expanded: false,
    ...overrides,
  };
}

function makeTrace(overrides: Partial<TraceDetailResponse> = {}): TraceDetailResponse {
  return {
    trace_id: '9f31c0aa',
    created_at: 0,
    question: 'How do I run it?',
    mode: 'stream',
    status: 'ok',
    config: null,
    candidates: [makeCandidate()],
    prompt_messages: null,
    answer: 'Run docker compose up [1].',
    cited_source_numbers: [1],
    timings: {
      embed_ms: 20,
      search_ms: 30,
      rerank_ms: 400,
      generate_ms: 1800,
      total_ms: 2310,
      condense_ms: 0,
      query_expansion_ms: 0,
      context_expansion_ms: 60,
    },
    error: null,
    ...overrides,
  };
}

function setup(props: Partial<Parameters<typeof Inspector>[0]> = {}) {
  const merged = {
    engineerMode: true,
    tab: 'sources' as const,
    onTabChange: vi.fn(),
    onClose: vi.fn(),
    sources: [makeSource()],
    traceId: 'trace-1',
    ...props,
  };
  render(<Inspector {...merged} />);
  return merged;
}

describe('Inspector', () => {
  beforeEach(() => {
    getTrace.mockReset();
    getTrace.mockResolvedValue(makeTrace());
    clearTraceCache();
  });

  it('shows only the Sources tab in reader mode', () => {
    setup({ engineerMode: false });

    expect(screen.getByTestId('inspector-tab-sources')).toBeInTheDocument();
    expect(screen.queryByTestId('inspector-tab-retrieval')).not.toBeInTheDocument();
    expect(screen.queryByTestId('inspector-tab-trace')).not.toBeInTheDocument();
  });

  it('adds Retrieval and Trace in engineer mode', () => {
    setup({ engineerMode: true });

    expect(screen.getByTestId('inspector-tab-retrieval')).toBeInTheDocument();
    expect(screen.getByTestId('inspector-tab-trace')).toBeInTheDocument();
  });

  it('falls back to Sources when leaving engineer mode on a hidden tab', () => {
    // Otherwise the panel is stranded on a tab its own tab bar no longer offers.
    const props = setup({ engineerMode: false, tab: 'trace' });

    expect(props.onTabChange).toHaveBeenCalledWith('sources');
    expect(screen.getByTestId('sources-tab')).toBeInTheDocument();
  });

  it('renders source cards with their rerank score once the trace loads', async () => {
    setup({ tab: 'sources' });

    expect(screen.getByTestId('sources-tab-card')).toHaveTextContent('guide.md');
    // The score comes from the trace candidate matched by chunk_id — citations carry none.
    expect(await screen.findByText('0.91')).toBeInTheDocument();
  });

  it('opens the source viewer when a source card is clicked', async () => {
    const onOpenSource = vi.fn();
    setup({ tab: 'sources', onOpenSource });

    await userEvent.click(screen.getByTestId('sources-tab-card'));

    expect(onOpenSource).toHaveBeenCalledWith('guide.md', 'chunk-a');
  });

  it('renders retrieval stat tiles and the candidate list', async () => {
    setup({ tab: 'retrieval' });

    expect(await screen.findByTestId('retrieval-tab')).toBeInTheDocument();
    expect(screen.getByText('2.31s')).toBeInTheDocument();
    expect(screen.getByText('1/1')).toBeInTheDocument();
    expect(screen.getByText('0.91')).toBeInTheDocument();
    expect(screen.getByTestId('retrieval-candidate')).toHaveTextContent('guide.md');
  });

  it('renders trace spans in pipeline order, skipping stages that did not run', async () => {
    setup({ tab: 'trace' });

    expect(await screen.findByTestId('trace-tab')).toBeInTheDocument();
    const spans = screen.getAllByTestId('trace-span');
    // condense and query expansion are 0ms here, so they are omitted rather than drawn
    // as zero-width bars. Context expansion sits AFTER rerank, not before embed.
    expect(spans.map((s) => s.textContent?.split('ms')[0])).toEqual([
      expect.stringContaining('embed'),
      expect.stringContaining('search'),
      expect.stringContaining('rerank'),
      expect.stringContaining('context expansion'),
      expect.stringContaining('generate'),
    ]);
  });

  it('degrades to an empty state when the trace was evicted', async () => {
    getTrace.mockRejectedValue(new ApiClientError('gone', 404));
    setup({ tab: 'trace' });

    expect(await screen.findByTestId('inspector-trace-unavailable')).toBeInTheDocument();
    expect(screen.queryByTestId('trace-tab')).not.toBeInTheDocument();
  });

  it('degrades to an empty state when the turn has no trace at all', async () => {
    // TRACE_ENABLED=false means the stream never carried a trace id.
    setup({ tab: 'retrieval', traceId: null });

    expect(await screen.findByTestId('inspector-trace-unavailable')).toBeInTheDocument();
    expect(getTrace).not.toHaveBeenCalled();
  });

  it('shows a pending state and fetches nothing while the turn is still streaming', () => {
    // Regression coverage for a real bug found via live Ollama generation: trace_id is
    // known (sent with the `sources` SSE event) well before generation finishes, but
    // the trace itself isn't written server-side until the terminal event. Fetching in
    // that gap 404s, and since traceId never changes across the gap, it would never
    // retry — permanently mislabeling a real trace as evicted.
    setup({ tab: 'retrieval', traceReady: false });

    expect(screen.getByTestId('inspector-trace-pending')).toBeInTheDocument();
    expect(getTrace).not.toHaveBeenCalled();
  });

  it('fetches once the turn finishes, using the same traceId that was already known', async () => {
    const { rerender } = render(
      <Inspector
        engineerMode
        tab="trace"
        onTabChange={vi.fn()}
        onClose={vi.fn()}
        sources={[makeSource()]}
        traceId="trace-1"
        traceReady={false}
      />,
    );
    expect(screen.getByTestId('inspector-trace-pending')).toBeInTheDocument();
    expect(getTrace).not.toHaveBeenCalled();

    rerender(
      <Inspector
        engineerMode
        tab="trace"
        onTabChange={vi.fn()}
        onClose={vi.fn()}
        sources={[makeSource()]}
        traceId="trace-1"
        traceReady
      />,
    );

    expect(await screen.findByTestId('trace-tab')).toBeInTheDocument();
    expect(getTrace).toHaveBeenCalledTimes(1);
  });

  it('still shows sources when the trace is unavailable', async () => {
    getTrace.mockRejectedValue(new ApiClientError('gone', 404));
    setup({ tab: 'sources' });

    // Sources come from the turn itself, so they must not depend on the trace.
    expect(screen.getByTestId('sources-tab-card')).toBeInTheDocument();
    await waitFor(() => expect(getTrace).toHaveBeenCalled());
    expect(screen.getByTestId('sources-tab-card')).toBeInTheDocument();
  });

  it('closes via the close button', async () => {
    const props = setup();

    await userEvent.click(screen.getByTestId('inspector-close'));

    expect(props.onClose).toHaveBeenCalled();
  });
});
