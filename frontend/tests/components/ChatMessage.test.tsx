import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';

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

describe('ChatMessage', () => {
  it('renders a user turn without citations', () => {
    render(<ChatMessage turn={makeTurn({ role: 'user', content: 'How do I run this?' })} />);

    const message = screen.getByTestId('chat-message');
    expect(message).toHaveAttribute('data-role', 'user');
    expect(screen.getByText('How do I run this?')).toBeInTheDocument();
    expect(screen.queryByTestId('citation-card')).not.toBeInTheDocument();
  });

  it('renders an assistant turn with citation cards hidden behind a sources toggle', async () => {
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
    expect(screen.queryByTestId('citation-card')).not.toBeInTheDocument();

    const toggle = screen.getByTestId('chat-message-sources-toggle');
    expect(toggle).toHaveTextContent('Sources (1)');

    await userEvent.click(toggle);

    expect(screen.getByTestId('citation-card')).toBeInTheDocument();
    expect(toggle).toHaveTextContent('Hide sources');

    await userEvent.click(toggle);

    expect(screen.queryByTestId('citation-card')).not.toBeInTheDocument();
    expect(toggle).toHaveTextContent('Sources (1)');
  });

  it('renders no sources toggle when the assistant turn has no sources', () => {
    render(<ChatMessage turn={makeTurn({ role: 'assistant', content: 'Answer.' })} />);

    expect(screen.queryByTestId('chat-message-sources-toggle')).not.toBeInTheDocument();
  });

  it('renders an error turn distinctly', () => {
    render(<ChatMessage turn={makeTurn({ role: 'error', content: 'API returned HTTP 502.' })} />);

    expect(screen.getByTestId('chat-message')).toHaveAttribute('data-role', 'error');
    expect(screen.getByText('API returned HTTP 502.')).toBeInTheDocument();
  });

  it('announces error turns to assistive tech via role="alert"', () => {
    render(<ChatMessage turn={makeTurn({ role: 'error', content: 'API returned HTTP 502.' })} />);

    expect(screen.getByRole('alert')).toHaveTextContent('API returned HTTP 502.');
  });

  it('renders the trace drawer toggle for an assistant turn with a traceId', () => {
    render(
      <ChatMessage turn={makeTurn({ role: 'assistant', content: 'Answer.', traceId: 'trace-1' })} />,
    );

    expect(screen.getByTestId('trace-drawer-toggle')).toBeInTheDocument();
  });

  it('renders no trace drawer when the turn has no traceId', () => {
    render(<ChatMessage turn={makeTurn({ role: 'assistant', content: 'Answer.' })} />);

    expect(screen.queryByTestId('trace-drawer-toggle')).not.toBeInTheDocument();
  });

  it('renders assistant markdown content (bold, lists) as real elements', () => {
    render(
      <ChatMessage
        turn={makeTurn({
          role: 'assistant',
          content: '**Important**\n\n- one\n- two',
        })}
      />,
    );

    expect(screen.getByText('Important').tagName).toBe('STRONG');
    expect(screen.getByText('one').closest('ul')).toBeInTheDocument();
    expect(screen.getByText('two').closest('li')).toBeInTheDocument();
  });

  it('does not inject raw HTML from assistant content', () => {
    render(
      <ChatMessage
        turn={makeTurn({ role: 'assistant', content: '<img src=x onerror=alert(1)>' })}
      />,
    );

    expect(document.querySelector('img')).not.toBeInTheDocument();
  });

  it('renders a timestamp for every turn', () => {
    render(<ChatMessage turn={makeTurn({ timestamp: Date.now() })} />);

    expect(document.querySelector('time')).toBeInTheDocument();
  });

  describe('copy button', () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('copies the assistant answer to the clipboard when clicked', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined);
      vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });

      render(<ChatMessage turn={makeTurn({ role: 'assistant', content: 'Answer text.' })} />);
      await userEvent.click(screen.getByTestId('chat-message-copy'));

      expect(writeText).toHaveBeenCalledWith('Answer text.');
      expect(await screen.findByText('Copied')).toBeInTheDocument();
    });

    it('renders no copy button when the clipboard API is unavailable', () => {
      vi.stubGlobal('navigator', { ...navigator, clipboard: undefined });

      render(<ChatMessage turn={makeTurn({ role: 'assistant', content: 'Answer text.' })} />);

      expect(screen.queryByTestId('chat-message-copy')).not.toBeInTheDocument();
    });
  });

  it('renders a Retry button on an error turn with a retained question and fires onRetry', async () => {
    const onRetry = vi.fn();
    render(
      <ChatMessage
        turn={makeTurn({ role: 'error', content: 'boom', question: 'What is DocRAG?' })}
        onRetry={onRetry}
      />,
    );

    await userEvent.click(screen.getByTestId('chat-message-retry'));

    expect(onRetry).toHaveBeenCalledWith('What is DocRAG?');
  });

  it('renders no Retry button when the error turn has no retained question', () => {
    render(<ChatMessage turn={makeTurn({ role: 'error', content: 'boom' })} onRetry={vi.fn()} />);

    expect(screen.queryByTestId('chat-message-retry')).not.toBeInTheDocument();
  });
});
