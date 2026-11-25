import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { ChatTurn } from '../../src/components/ChatMessage';
import SavedPanel from '../../src/components/SavedPanel';
import { recheckRequest, recheckSummary } from '../../src/utils/savedCheck';

const turn: ChatTurn = {
  id: 'a1',
  role: 'assistant',
  content: 'Staff get 28 days [1].',
  sources: [{ source_number: 1, filename: 'leave.txt', page: 1, section: 'Leave', chunk_id: 'c1', text: 'x' }],
  timestamp: 1,
  timings: null,
  traceId: 't1',
  saved: true,
};
const item = { conversationId: 'c', conversationTitle: 'Leave', workspace: '', turn, question: 'Leave?' };

afterEach(() => vi.restoreAllMocks());

describe('pass 21: recheck a saved answer', () => {
  it('sends the answer and its cited passages', () => {
    expect(recheckRequest(item)).toEqual({
      answer: 'Staff get 28 days [1].',
      sources: [{ source_number: 1, filename: 'leave.txt', chunk_id: 'c1' }],
    });
  });

  it('says in one line whether it still holds', () => {
    expect(recheckSummary({ sources: [], sentences: [], still_holds: true })).toMatch(/^Still holds/);
    expect(
      recheckSummary({
        sources: [
          { source_number: 1, filename: 'a', chunk_id: 'c1', state: 'changed' },
          { source_number: 2, filename: 'b', chunk_id: 'c2', state: 'deleted' },
        ],
        sentences: [{ sentence: 'x', status: 'numbers_differ' }],
        still_holds: false,
      }),
    ).toBe('Check it: 1 sentence no longer backed, 1 cited passage changed, 1 cited document deleted.');
  });

  it('rechecks one answer from the Saved panel', async () => {
    const recheck = vi
      .spyOn(api, 'recheckAnswer')
      .mockResolvedValue({ sources: [], sentences: [], still_holds: true });
    render(<SavedPanel answers={[item]} onOpen={vi.fn()} onUnsave={vi.fn()} workspace="" />);
    await userEvent.click(screen.getByTestId('saved-recheck'));
    await waitFor(() => expect(screen.getByTestId('saved-recheck-result')).toHaveTextContent('Still holds'));
    expect(recheck).toHaveBeenCalledWith(recheckRequest(item), expect.any(AbortSignal), '');
  });
});
