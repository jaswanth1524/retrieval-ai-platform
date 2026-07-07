import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ChatThread from '../../src/components/ChatThread';
import type { ChatTurn } from '../../src/components/ChatMessage';

function makeTurn(overrides: Partial<ChatTurn>): ChatTurn {
  return {
    id: 'turn-1',
    role: 'user',
    content: 'Hello',
    sources: [],
    timestamp: 0,
    ...overrides,
  };
}

describe('ChatThread', () => {
  it('shows an empty-state prompt when there are no turns and nothing is pending', () => {
    render(<ChatThread turns={[]} pending={false} onCancel={vi.fn()} />);

    expect(screen.getByText('Upload a document, then ask a question about it.')).toBeInTheDocument();
  });

  it('renders each turn in order', () => {
    const turns = [makeTurn({ id: 'a', content: 'First' }), makeTurn({ id: 'b', content: 'Second' })];
    render(<ChatThread turns={turns} pending={false} onCancel={vi.fn()} />);

    const messages = screen.getAllByTestId('chat-message');
    expect(messages).toHaveLength(2);
    expect(messages[0]).toHaveTextContent('First');
    expect(messages[1]).toHaveTextContent('Second');
  });

  it('shows the pending indicator, announced via role="status", while generating', () => {
    render(<ChatThread turns={[]} pending onCancel={vi.fn()} />);

    expect(screen.getByTestId('chat-pending')).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('Generating answer...');
  });

  it('does not show the empty-state prompt while pending with no turns yet', () => {
    render(<ChatThread turns={[]} pending onCancel={vi.fn()} />);

    expect(
      screen.queryByText('Upload a document, then ask a question about it.'),
    ).not.toBeInTheDocument();
  });

  it('calls onCancel when the Stop button is clicked while pending', async () => {
    const onCancel = vi.fn();
    render(<ChatThread turns={[]} pending onCancel={onCancel} />);

    await userEvent.click(screen.getByTestId('chat-cancel'));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('does not show a cancel button when nothing is pending', () => {
    render(<ChatThread turns={[makeTurn({})]} pending={false} onCancel={vi.fn()} />);

    expect(screen.queryByTestId('chat-cancel')).not.toBeInTheDocument();
  });
});
