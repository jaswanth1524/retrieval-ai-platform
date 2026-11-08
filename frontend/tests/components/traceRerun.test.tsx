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
});
