import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import { ScoreReportView } from '../../src/components/TraceTools';

const REPORT = {
  current_min_score: 0.15,
  bin_edges: [0, 1],
  answered: [25],
  unanswered: [25],
  answered_count: 25,
  unanswered_count: 25,
  suggested_min_score: 0.3,
  note: 'from 50 traced questions',
  would_change: 4,
};

afterEach(() => vi.restoreAllMocks());

describe('pass 21: the suggested bar applied', () => {
  it('is worked out and applied by the server, with the labels shown', async () => {
    vi.spyOn(api, 'scoreReport').mockResolvedValue(REPORT as never);
    const calibrate = vi
      .spyOn(api, 'calibrate')
      .mockResolvedValue({ ...REPORT, suggested_min_score: 0.32, applied: true } as never);
    const put = vi.spyOn(api, 'putWorkspaceSettings');
    render(<ScoreReportView onClose={() => {}} canApply />);
    await userEvent.click(await screen.findByTestId('score-report-apply'));
    expect(calibrate).toHaveBeenCalledWith({ apply: true, labels: 'model' });
    expect(put).not.toHaveBeenCalled();
    expect(await screen.findByTestId('score-report-applied')).toHaveTextContent('now uses 0.32');
  });
});
