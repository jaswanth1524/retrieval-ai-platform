import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ConversationList from '../../src/components/ConversationList';
import type { ConversationSummary } from '../../src/hooks/useChat';

const CONVERSATIONS: ConversationSummary[] = [
  { id: 'c1', title: 'First chat', updatedAt: 2, turnCount: 4 },
  { id: 'c2', title: 'Second chat', updatedAt: 1, turnCount: 9 },
];

function setup(overrides: Partial<Parameters<typeof ConversationList>[0]> = {}) {
  const props = {
    conversations: CONVERSATIONS,
    activeId: 'c1',
    onSwitch: vi.fn(),
    onRename: vi.fn(),
    onDelete: vi.fn(),
    ...overrides,
  };
  render(<ConversationList {...props} />);
  return props;
}

describe('ConversationList', () => {
  it('renders a row per conversation', () => {
    setup();
    expect(screen.getAllByTestId('conversation-item')).toHaveLength(2);
  });

  it('renders the turn count and marks the active conversation', () => {
    setup();
    const rows = screen.getAllByTestId('conversation-select');
    expect(rows[0]).toHaveTextContent('4 turns');
    expect(rows[1]).toHaveTextContent('9 turns');
    expect(rows[0].closest('.conversation-list__row')).toHaveClass('conversation-list__row--active');
  });

  it('fires onSwitch with the conversation id', async () => {
    const props = setup();
    await userEvent.click(screen.getByTitle('Second chat'));
    expect(props.onSwitch).toHaveBeenCalledWith('c2');
  });

  it('deleting requires an inline confirm before onDelete fires', async () => {
    const props = setup();
    await userEvent.click(screen.getByLabelText('Delete First chat'));
    expect(props.onDelete).not.toHaveBeenCalled();
    expect(screen.getByText('Delete this conversation?')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('conversation-delete-confirm'));
    expect(props.onDelete).toHaveBeenCalledWith('c1');
  });

  it('cancelling the delete confirm does not call onDelete', async () => {
    const props = setup();
    await userEvent.click(screen.getByLabelText('Delete First chat'));
    await userEvent.click(screen.getByText('Cancel'));

    expect(props.onDelete).not.toHaveBeenCalled();
    expect(screen.queryByText('Delete this conversation?')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete First chat')).toBeInTheDocument();
  });

  it('dismisses the delete confirm on Escape', async () => {
    const props = setup();
    await userEvent.click(screen.getByLabelText('Delete First chat'));
    // Cancel takes focus on open, so Escape reaches the handler without tabbing.
    await userEvent.keyboard('{Escape}');

    expect(props.onDelete).not.toHaveBeenCalled();
    expect(screen.queryByText('Delete this conversation?')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete First chat')).toBeInTheDocument();
  });

  it('clears a pending delete confirm when the active conversation changes', async () => {
    const handlers = { onSwitch: vi.fn(), onRename: vi.fn(), onDelete: vi.fn() };
    const { rerender } = render(
      <ConversationList conversations={CONVERSATIONS} activeId="c1" {...handlers} />,
    );
    // Arm the confirmation, then switch conversations. Left armed, the row keeps its
    // prompt and stays unselectable (the confirm block replaces its switch button).
    await userEvent.click(screen.getByLabelText('Delete First chat'));
    expect(screen.getByText('Delete this conversation?')).toBeInTheDocument();

    rerender(<ConversationList conversations={CONVERSATIONS} activeId="c2" {...handlers} />);

    expect(screen.queryByText('Delete this conversation?')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete First chat')).toBeInTheDocument();
    expect(handlers.onDelete).not.toHaveBeenCalled();
  });

  it('keeps the confirm armed when an unrelated turn updates the list', async () => {
    const base = { activeId: 'c1', onSwitch: vi.fn(), onRename: vi.fn(), onDelete: vi.fn() };
    const { rerender } = render(<ConversationList conversations={CONVERSATIONS} {...base} />);
    await userEvent.click(screen.getByLabelText('Delete First chat'));
    expect(screen.getByText('Delete this conversation?')).toBeInTheDocument();

    // New array identity, same ids — what an appended turn or a rename produces. The
    // prompt must survive this, or unrelated activity yanks it away mid-decision.
    rerender(
      <ConversationList
        conversations={[{ id: 'c1', title: 'First chat', updatedAt: 9, turnCount: 5 }, CONVERSATIONS[1]]}
        {...base}
      />,
    );

    expect(screen.getByText('Delete this conversation?')).toBeInTheDocument();
  });

  it('renames via the inline edit field on Enter', async () => {
    const props = setup();
    await userEvent.click(screen.getByLabelText('Rename First chat'));
    const input = screen.getByLabelText('Rename First chat');
    await userEvent.clear(input);
    await userEvent.type(input, 'Renamed{Enter}');
    expect(props.onRename).toHaveBeenCalledWith('c1', 'Renamed');
  });
});
