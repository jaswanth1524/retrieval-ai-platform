import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { TraceDetailResponse, TraceRerunResponse } from '../../src/api/types';
import TraceRerun from '../../src/components/inspector/TraceRerun';

const trace = {
  trace_id: 't1',
  kind: 'question',
  config: { rerank_min_score: 0.15, rerank_top_k: 8 },
} as unknown as TraceDetailResponse;

afterEach(() => vi.restoreAllMocks());

describe('TraceRerun', () => {
  it('names the section of each moved passage, so two on one page read apart', async () => {
    const result = {
      query: 'q',
      settings: { rerank_min_score: 0.99 },
      would_answer: false,
      weak_matches: true,
      diff: {
        question_a: 'q',
        question_b: 'q',
        same_answer: true,
        settings: [],
        candidates: [
          { point_id: 'p1', filename: 'leave.md', page: 1, section: 'About this policy', chunk_id: 'c1', rank_a: 3, rank_b: 3, selected_a: false, selected_b: true },
          { point_id: 'p2', filename: 'old.md', page: 2, chunk_id: 'c2', rank_a: 1, rank_b: 2 },
        ],
      },
    } as unknown as TraceRerunResponse;
    vi.spyOn(api, 'rerunTrace').mockResolvedValue(result);
    render(<TraceRerun trace={trace} />);
    await userEvent.click(screen.getByTestId('trace-rerun-submit'));
    const shown = await screen.findByTestId('trace-rerun-result');
    expect(shown).toHaveTextContent('leave.md p.1 · About this policy: rank 3 → 3');
    // An older server sends no section: the label is as before.
    expect(shown).toHaveTextContent('old.md p.2: rank 1 → 2');
  });

  it('a search still running when the trace goes away is aborted, not shown', async () => {
    let seen: AbortSignal | undefined;
    vi.spyOn(api, 'rerunTrace').mockImplementation((_id, _settings, signal) => {
      seen = signal;
      return new Promise(() => {});
    });
    const { unmount } = render(<TraceRerun trace={trace} />);
    await userEvent.click(screen.getByTestId('trace-rerun-submit'));
    expect(seen?.aborted).toBe(false);
    unmount();
    expect(seen?.aborted).toBe(true);
  });
});

describe('score report what-if', () => {
  it('tries a bar on the latest questions and says how many would be answered', async () => {
    const { ScoreReportView } = await import('../../src/components/TraceTools');
    vi.spyOn(api, 'scoreReport').mockResolvedValue({
      current_min_score: 0.15,
      bin_edges: [0, 1],
      answered: [1],
      unanswered: [1],
      answered_count: 1,
      unanswered_count: 1,
      suggested_min_score: null,
      note: 'n',
      would_change: 0,
    } as never);
    const tried = vi.spyOn(api, 'traceWhatIf').mockResolvedValue({
      settings: { rerank_min_score: 0.1 },
      checked: 5,
      would_answer: 3,
      newly_answered: 3,
      newly_unanswered: 0,
      items: [],
    });
    render(<ScoreReportView onClose={() => {}} />);
    const input = await screen.findByTestId('what-if-bar');
    await userEvent.clear(input);
    await userEvent.type(input, '0.1');
    await userEvent.click(screen.getByTestId('what-if-run'));
    expect(await screen.findByTestId('what-if-result')).toHaveTextContent('3 of 5 would be answered · 3 newly');
    expect(tried.mock.calls[0][0]).toEqual({ rerank_min_score: 0.1, limit: 20, only_unanswered: true });
  });
});
