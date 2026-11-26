import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ConversationList from '../../src/components/ConversationList';
import type { ConversationSummary } from '../../src/hooks/useChat';

const CONVERSATIONS: ConversationSummary[] = [
  { id: 'a', title: 'Alpha', updatedAt: 3, turnCount: 2 },
  { id: 'b', title: 'Beta', updatedAt: 2, turnCount: 2, labels: ['hr'] },
  { id: 'c', title: 'Gamma', updatedAt: 1, turnCount: 2, pinned: true },
];

function setup() {
  const props = {
    conversations: CONVERSATIONS,
    activeId: 'a',
    onSwitch: vi.fn(),
    onRename: vi.fn(),
    onDelete: vi.fn(),
    onDeleteMany: vi.fn(),
    onPin: vi.fn(),
    onSetLabels: vi.fn(),
  };
  render(<ConversationList {...props} />);
  return props;
}

async function pick(...titles: string[]) {
  await userEvent.click(screen.getByTestId('conversation-select-mode'));
  for (const title of titles) {
    await userEvent.click(screen.getByRole('checkbox', { name: `Select ${title}` }));
  }
}

describe('pass 21: selecting several conversations', () => {
  it('deletes the picked ones after one confirmation', async () => {
    const props = setup();
    await pick('Alpha', 'Beta');
    expect(screen.getByTestId('conversation-bulk')).toHaveTextContent('2 selected');
    await userEvent.click(screen.getByTestId('conversation-bulk-delete'));
    await userEvent.click(screen.getByTestId('conversation-bulk-delete-confirm'));
    expect(props.onDeleteMany).toHaveBeenCalledWith(['a', 'b']);
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
  });

  it('pins them, or unpins them when all are pinned', async () => {
    const props = setup();
    await pick('Beta', 'Gamma');
    await userEvent.click(screen.getByTestId('conversation-bulk-pin'));
    // Gamma is pinned already: only Beta needs it (pass 22: within the cap).
    expect(props.onPin.mock.calls).toEqual([['b', true]]);
  });

  it('adds a label to each, keeping their own', async () => {
    const props = setup();
    await pick('Alpha', 'Beta');
    await userEvent.click(screen.getByTestId('conversation-bulk-label'));
    await userEvent.type(screen.getByTestId('conversation-bulk-label-input'), 'audit{Enter}');
    expect(props.onSetLabels.mock.calls).toEqual([
      ['a', ['audit']],
      ['b', ['hr', 'audit']],
    ]);
  });

  it('selects all shown with All, and leaves with Done', async () => {
    setup();
    await pick();
    await userEvent.click(screen.getByTestId('conversation-select-all'));
    expect(within(screen.getByTestId('conversation-bulk')).getByText('3 selected')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('conversation-select-done'));
    expect(screen.getByTestId('conversation-select-mode')).toBeInTheDocument();
  });
});
