import { act, render, renderHook, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { RecheckResponse } from '../../src/api/types';
import type { ChatTurn } from '../../src/components/ChatMessage';
import ConversationList from '../../src/components/ConversationList';
import SavedPanel from '../../src/components/SavedPanel';
import type { ConversationSummary } from '../../src/hooks/useChat';
import type { SourceView } from '../../src/hooks/useLinks';
import { useViewerTrail } from '../../src/hooks/useViewerTrail';
import { markUnbacked, UNBACKED_HREF } from '../../src/utils/citations';
import { recheckSummary } from '../../src/utils/savedCheck';

afterEach(() => vi.restoreAllMocks());

function saved(id: string, content: string) {
  const turn: ChatTurn = {
    id,
    role: 'assistant',
    content,
    sources: [{ source_number: 1, filename: 'leave.txt', page: 1, section: 'Leave', chunk_id: 'c1', text: 'x' }],
    timestamp: 1,
    timings: null,
    traceId: null,
    saved: true,
  };
  return { conversationId: 'c', conversationTitle: `T ${id}`, workspace: '', turn, question: 'Leave?' };
}

describe('pass 22: rechecks of two saved answers', () => {
  it('each finishes, though the second began before the first returned', async () => {
    const replies: Array<(value: RecheckResponse) => void> = [];
    vi.spyOn(api, 'recheckAnswer').mockImplementation(
      () => new Promise<RecheckResponse>((resolve) => replies.push(resolve)),
    );
    render(
      <SavedPanel
        answers={[saved('a1', 'Staff get 28 days [1].'), saved('a2', 'Staff get 30 days [1].')]}
        onOpen={vi.fn()}
        onUnsave={vi.fn()}
        workspace=""
      />,
    );
    const buttons = screen.getAllByTestId('saved-recheck');
    await userEvent.click(buttons[0]);
    await userEvent.click(buttons[1]);
    act(() => {
      replies[0]({ sources: [], sentences: [], still_holds: true });
      replies[1]({ sources: [], sentences: [], still_holds: true });
    });
    await waitFor(() =>
      expect(screen.getAllByTestId('saved-recheck-result').map((e) => e.textContent)).toEqual([
        expect.stringMatching(/^Still holds/),
        expect.stringMatching(/^Still holds/),
      ]),
    );
  });

  it('says what held when only the first sentences were checked', () => {
    expect(
      recheckSummary({
        sources: [],
        sentences: Array.from({ length: 20 }, () => ({ sentence: 'x', status: 'supported' })),
        still_holds: false,
        more: true,
      }),
    ).toMatch(/^Holds so far: its first 20 sentences/);
  });
});

const CONVERSATIONS: ConversationSummary[] = [
  { id: 'a', title: 'Alpha', updatedAt: 3, turnCount: 2 },
  { id: 'b', title: 'Beta', updatedAt: 2, turnCount: 2 },
  { id: 'c', title: 'Gamma', updatedAt: 1, turnCount: 2 },
];

describe('pass 22: bulk actions', () => {
  it('act on the picked rows still shown', async () => {
    const onDeleteMany = vi.fn();
    // Enough rows for the search box to show.
    const many: ConversationSummary[] = [
      ...CONVERSATIONS,
      ...Array.from({ length: 10 }, (_, i) => ({
        id: `x${i}`,
        title: `Other ${i}`,
        updatedAt: 0,
        turnCount: 2,
      })),
    ];
    render(
      <ConversationList
        conversations={many}
        activeId="a"
        onSwitch={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        onDeleteMany={onDeleteMany}
        onPin={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId('conversation-select-mode'));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Beta' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Gamma' }));
    await userEvent.type(screen.getByTestId('conversation-search'), 'Gam');
    await waitFor(() =>
      expect(within(screen.getByTestId('conversation-bulk')).getByText('1 selected')).toBeInTheDocument(),
    );
    await userEvent.click(screen.getByTestId('conversation-bulk-delete'));
    await userEvent.click(screen.getByTestId('conversation-bulk-delete-confirm'));
    expect(onDeleteMany).toHaveBeenCalledWith(['c']);
  });

  it('pin no more than there is room for', async () => {
    const pinnedNine: ConversationSummary[] = [
      ...Array.from({ length: 9 }, (_, i) => ({
        id: `p${i}`,
        title: `Pinned ${i}`,
        updatedAt: 100 + i,
        turnCount: 2,
        pinned: true,
      })),
      ...CONVERSATIONS,
    ];
    const onPin = vi.fn(() => true);
    render(
      <ConversationList
        conversations={pinnedNine}
        activeId="a"
        onSwitch={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        onPin={onPin}
      />,
    );
    await userEvent.click(screen.getByTestId('conversation-select-mode'));
    for (const title of ['Alpha', 'Beta', 'Gamma']) {
      await userEvent.click(screen.getByRole('checkbox', { name: `Select ${title}` }));
    }
    await userEvent.click(screen.getByTestId('conversation-bulk-pin'));
    expect(onPin.mock.calls).toEqual([['a', true]]);
  });
});

describe('pass 22: self-check marks', () => {
  const sentence = 'Staff take thirty days of leave';
  it('are not put inside code, tables or links', () => {
    for (const markdown of [
      `\`\`\`\n${sentence}\n\`\`\``,
      `Run \`${sentence}\` now.`,
      `| a | ${sentence} |`,
      `[${sentence}](https://example.com)`,
    ]) {
      expect(markUnbacked(markdown, [sentence])).toBe(markdown);
    }
    expect(markUnbacked(`${sentence} [1].`, [sentence])).toBe(`[${sentence}](${UNBACKED_HREF}) [1].`);
  });
});

describe('pass 22: viewer trail', () => {
  it('two Backs in a row land two places back', () => {
    const { result } = renderHook(() => {
      const [view, setView] = useState<SourceView | null>(null);
      return { view, setView, trail: useViewerTrail(view, setView) };
    });
    const place = (chunkId: string): SourceView => ({ filename: 'a.txt', chunkId });
    act(() => result.current.setView(place('c1')));
    act(() => result.current.setView(place('c2')));
    act(() => result.current.setView(place('c3')));
    act(() => {
      result.current.trail.back?.();
      result.current.trail.back?.();
    });
    expect(result.current.view?.chunkId).toBe('c1');
    act(() => result.current.trail.forward?.());
    expect(result.current.view?.chunkId).toBe('c2');
    act(() => result.current.trail.forward?.());
    expect(result.current.view?.chunkId).toBe('c3');
  });
});

describe('pass 22: a flagged sentence in the answer', () => {
  it('opens the self-check at its own line when clicked', async () => {
    const { default: ChatMessage } = await import('../../src/components/ChatMessage');
    const turn: ChatTurn = {
      id: 'a1',
      role: 'assistant',
      content: 'Staff get thirty days of leave [1]. The moon is made of cheese [1].',
      sources: [],
      timestamp: 1,
      timings: null,
      traceId: 't1',
      flaggedSentences: [
        { sentence: 'Staff get thirty days of leave [1].', status: 'unsupported' },
        { sentence: 'The moon is made of cheese [1].', status: 'unsupported' },
      ],
    };
    render(<ChatMessage turn={turn} engineerMode={false} />);
    const marks = await screen.findAllByTestId('answer-unbacked');
    expect(marks[1]).toHaveTextContent('(not backed by its cited passages)');
    await userEvent.click(within(marks[1]).getByRole('button'));
    const details = screen.getByTestId('answer-self-check') as HTMLDetailsElement;
    expect(details.open).toBe(true);
    expect(document.activeElement?.textContent).toContain('The moon is made of cheese');
  });
});
