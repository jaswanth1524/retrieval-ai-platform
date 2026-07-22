import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ConversationList from '../../src/components/ConversationList';
import type { ConversationSummary } from '../../src/hooks/useChat';

const CONVERSATIONS: ConversationSummary[] = [
  { id: 'c1', title: 'First chat', updatedAt: 2 },
  { id: 'c2', title: 'Second chat', updatedAt: 1 },
];

function setup(overrides: Partial<Parameters<typeof ConversationList>[0]> = {}) {
  const props = {
    conversations: CONVERSATIONS,
    activeId: 'c1',
    onNew: vi.fn(),
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

  it('fires onNew when New chat is clicked', async () => {
    const props = setup();
    await userEvent.click(screen.getByTestId('conversation-new'));
    expect(props.onNew).toHaveBeenCalledOnce();
  });

  it('fires onSwitch with the conversation id', async () => {
    const props = setup();
    await userEvent.click(screen.getByRole('button', { name: 'Second chat' }));
    expect(props.onSwitch).toHaveBeenCalledWith('c2');
  });

  it('fires onDelete with the conversation id', async () => {
    const props = setup();
    await userEvent.click(screen.getByLabelText('Delete First chat'));
    expect(props.onDelete).toHaveBeenCalledWith('c1');
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
