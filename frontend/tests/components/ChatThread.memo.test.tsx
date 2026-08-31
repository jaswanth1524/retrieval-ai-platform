import { render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ChatThread from '../../src/components/ChatThread';
import type { ChatTurn } from '../../src/components/ChatMessage';

// Counts renders of the markdown body, keyed by its content — the streaming turn's
// content changes every delta (a new key each time, which is expected and not
// asserted on); prior turns' content is fixed, so their key's count is the real signal.
const renderCounts = new Map<string, number>();

vi.mock('react-markdown', () => ({
  default: ({ children }: { children: string }) => {
    renderCounts.set(children, (renderCounts.get(children) ?? 0) + 1);
    return <div data-testid="markdown-stub">{children}</div>;
  },
}));

function makeTurn(overrides: Partial<ChatTurn>): ChatTurn {
  return {
    id: overrides.id ?? 'turn',
    role: 'assistant',
    content: '',
    sources: [],
    timestamp: 0,
    timings: null,
    traceId: null,
    ...overrides,
  };
}

describe('ChatThread memoization', () => {
  beforeEach(() => renderCounts.clear());
  afterEach(() => vi.clearAllMocks());

  it('does not re-render prior turns on every delta of the currently streaming turn', () => {
    // Regression guard for A5: without React.memo(ChatMessage) plus stable callback
    // props (see ChatMessage.tsx and App.tsx's askQuestion/handleOpenSource/
    // handleCitationLeave), every SSE delta rebuilding the turns array re-renders
    // every prior ChatMessage — including a full ReactMarkdown reparse of answers
    // that haven't changed. Per-delta cost should scale with O(1), not conversation
    // length.
    const priorTurns: ChatTurn[] = Array.from({ length: 5 }, (_, i) =>
      makeTurn({ id: `prior-${i}`, role: 'assistant', content: `Answer number ${i}` }),
    );
    const streamingId = 'streaming';

    const buildTurns = (streamingContent: string): ChatTurn[] => [
      ...priorTurns,
      makeTurn({ id: streamingId, role: 'assistant', content: streamingContent }),
    ];

    const { rerender } = render(
      <ChatThread turns={buildTurns('')} pending engineerMode={false} />,
    );

    const deltas = ['Hel', 'Hello', 'Hello, ', 'Hello, world', 'Hello, world!'];
    for (const content of deltas) {
      rerender(<ChatThread turns={buildTurns(content)} pending engineerMode={false} />);
    }

    for (const turn of priorTurns) {
      expect(renderCounts.get(turn.content)).toBe(1);
    }
  });
});
