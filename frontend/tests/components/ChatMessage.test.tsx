import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';

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

describe('ChatMessage', () => {
  it('renders a user turn without citations', () => {
    render(<ChatMessage turn={makeTurn({ role: 'user', content: 'How do I run this?' })} />);

    const message = screen.getByTestId('chat-message');
    expect(message).toHaveAttribute('data-role', 'user');
    expect(screen.getByText('How do I run this?')).toBeInTheDocument();
    expect(screen.queryByTestId('citation-card')).not.toBeInTheDocument();
  });

  it('renders an assistant turn with citation cards', () => {
    render(
      <ChatMessage
        turn={makeTurn({
          role: 'assistant',
          content: 'Start Docker Compose [1].',
          sources: [
            {
              source_number: 1,
              filename: 'guide.md',
              page: 1,
              section: 'Setup',
              chunk_id: 'abc123',
              text: 'Start Docker Compose which launches Qdrant, the API, and the UI.',
            },
          ],
        })}
      />,
    );

    const message = screen.getByTestId('chat-message');
    expect(message).toHaveAttribute('data-role', 'assistant');
    expect(screen.getByTestId('citation-card')).toBeInTheDocument();
  });

  it('renders an error turn distinctly', () => {
    render(<ChatMessage turn={makeTurn({ role: 'error', content: 'API returned HTTP 502.' })} />);

    expect(screen.getByTestId('chat-message')).toHaveAttribute('data-role', 'error');
    expect(screen.getByText('API returned HTTP 502.')).toBeInTheDocument();
  });
});
