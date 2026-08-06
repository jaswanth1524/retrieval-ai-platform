import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ChatThread from '../../src/components/ChatThread';
import type { ChatTurn } from '../../src/components/ChatMessage';

function makeTurn(overrides: Partial<ChatTurn>): ChatTurn {
  return {
    id: 'turn-1',
    role: 'user',
    content: 'Hello',
    sources: [],
    timestamp: 0,
    timings: null,
    traceId: null,
    ...overrides,
  };
}

describe('ChatThread', () => {
  it('shows an empty-state prompt when there are no turns and nothing is pending', () => {
    render(<ChatThread turns={[]} pending={false} engineerMode={false} />);

    expect(screen.getByText('Upload a document, then ask a question about it.')).toBeInTheDocument();
  });

  it('renders each turn in order', () => {
    const turns = [makeTurn({ id: 'a', content: 'First' }), makeTurn({ id: 'b', content: 'Second' })];
    render(<ChatThread turns={turns} pending={false} engineerMode={false} />);

    const messages = screen.getAllByTestId('chat-message');
    expect(messages).toHaveLength(2);
    expect(messages[0]).toHaveTextContent('First');
    expect(messages[1]).toHaveTextContent('Second');
  });

  it('shows a placeholder skeleton in the "retrieving…" stage before the assistant turn exists', () => {
    render(<ChatThread turns={[makeTurn({ role: 'user' })]} pending engineerMode={false} />);

    expect(screen.getByTestId('chat-pending')).toBeInTheDocument();
    expect(screen.getByTestId('streaming-skeleton')).toHaveTextContent('retrieving…');
  });

  it('does not show the empty-state prompt while pending with no turns yet', () => {
    render(<ChatThread turns={[]} pending engineerMode={false} />);

    expect(
      screen.queryByText('Upload a document, then ask a question about it.'),
    ).not.toBeInTheDocument();
  });

  it('passes a "generating answer…" stream stage to the last turn once the assistant turn exists', () => {
    const turns = [
      makeTurn({ id: 'q', role: 'user', content: 'Question' }),
      makeTurn({ id: 'a', role: 'assistant', content: '' }),
    ];
    render(<ChatThread turns={turns} pending engineerMode={false} />);

    expect(screen.queryByTestId('chat-pending')).not.toBeInTheDocument();
    expect(screen.getByTestId('streaming-skeleton')).toHaveTextContent('generating answer…');
  });

  it('does not pass a stream stage to any turn when nothing is pending', () => {
    const turns = [makeTurn({ id: 'a', role: 'assistant', content: '' })];
    render(<ChatThread turns={turns} pending={false} engineerMode={false} />);

    expect(screen.queryByTestId('streaming-skeleton')).not.toBeInTheDocument();
  });
});
