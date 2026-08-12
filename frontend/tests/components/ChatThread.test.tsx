import { fireEvent, render, screen } from '@testing-library/react';
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

  /** jsdom reports 0 for every layout metric, so the scroll container's geometry has to
   *  be faked for any scroll-position assertion to mean anything. */
  function stubGeometry(element: HTMLElement, scrollHeight: number, clientHeight: number): void {
    Object.defineProperty(element, 'scrollHeight', { value: scrollHeight, configurable: true });
    Object.defineProperty(element, 'clientHeight', { value: clientHeight, configurable: true });
  }

  /** Move the scroll position the way a user would, then notify the component.
   *
   *  The component tells its own autoscroll apart from a user's by position, so a test
   *  simulating a user has to actually land somewhere the component did not park it. */
  function userScrollTo(element: HTMLElement, top: number): void {
    element.scrollTop = top;
    fireEvent.scroll(element);
  }

  it('keeps scrolling to the bottom as the streaming answer grows', () => {
    // The turn count never changes between the first delta and `done` — the whole point
    // of the bug this covers. Only the assistant turn's content grows.
    const question = makeTurn({ id: 'q', role: 'user', content: 'Question' });
    const { rerender } = render(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'Line one.' })]}
        pending
        engineerMode={false}
      />,
    );

    const thread = screen.getByTestId('chat-thread');
    stubGeometry(thread, 2000, 400);

    rerender(
      <ChatThread
        turns={[
          question,
          makeTurn({ id: 'a', role: 'assistant', content: 'Line one. Line two. Line three.' }),
        ]}
        pending
        engineerMode={false}
      />,
    );

    expect(thread.scrollTop).toBe(2000);
  });

  it('does not unpin itself when its own autoscroll fires a scroll event', () => {
    // Regression guard for a bug a real browser found and jsdom could not: jsdom never
    // dispatches `scroll` when scrollTop is assigned, so the component's own scroll used
    // to unpin it — the effect scrolled against the height it could see, the next delta
    // grew the content, and the queued scroll event measured a gap against the NEW
    // height and concluded the user had scrolled away. scrollTop then froze mid-answer.
    const question = makeTurn({ id: 'q', role: 'user', content: 'Question' });
    const { rerender } = render(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'One.' })]}
        pending
        engineerMode={false}
      />,
    );

    const thread = screen.getByTestId('chat-thread');
    stubGeometry(thread, 1000, 400);

    rerender(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'One. Two.' })]}
        pending
        engineerMode={false}
      />,
    );
    expect(thread.scrollTop).toBe(1000);

    // The content grows, THEN the scroll event from our own assignment lands — the exact
    // ordering that used to unpin. The listener must ignore it.
    stubGeometry(thread, 1600, 400);
    fireEvent.scroll(thread);

    rerender(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'One. Two. Three.' })]}
        pending
        engineerMode={false}
      />,
    );

    expect(thread.scrollTop).toBe(1600);
  });

  it('stops following the bottom once the user scrolls up to re-read', () => {
    const question = makeTurn({ id: 'q', role: 'user', content: 'Question' });
    const { rerender } = render(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'Line one.' })]}
        pending
        engineerMode={false}
      />,
    );

    const thread = screen.getByTestId('chat-thread');
    stubGeometry(thread, 2000, 400);
    // Scrolled well clear of the bottom (2000 - 200 - 400 = 1400px away).
    userScrollTo(thread, 200);

    rerender(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'Line one. More.' })]}
        pending
        engineerMode={false}
      />,
    );

    expect(thread.scrollTop).toBe(200);
  });

  it('re-pins to the bottom when a new turn arrives, even after scrolling up', () => {
    // Asking a new question is an explicit signal the user wants to see what happens
    // next, so it overrides an earlier scroll-up.
    const question = makeTurn({ id: 'q', role: 'user', content: 'Question' });
    const { rerender } = render(
      <ChatThread
        turns={[question, makeTurn({ id: 'a', role: 'assistant', content: 'Answer.' })]}
        pending={false}
        engineerMode={false}
      />,
    );

    const thread = screen.getByTestId('chat-thread');
    stubGeometry(thread, 2000, 400);
    userScrollTo(thread, 200);

    rerender(
      <ChatThread
        turns={[
          question,
          makeTurn({ id: 'a', role: 'assistant', content: 'Answer.' }),
          makeTurn({ id: 'q2', role: 'user', content: 'Follow-up' }),
        ]}
        pending
        engineerMode={false}
      />,
    );

    expect(thread.scrollTop).toBe(2000);
  });
});
