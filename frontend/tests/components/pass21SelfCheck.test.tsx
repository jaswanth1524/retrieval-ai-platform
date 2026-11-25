import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { flaggedFrom } from '../../src/api/client';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';
import { sanitizeTurn } from '../../src/hooks/chat/sanitize';

function answer(overrides: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id: 'a1',
    role: 'assistant',
    content: 'Staff get 30 days [1]. The moon is cheese [1].',
    sources: [],
    timestamp: 1_700_000_000_000,
    timings: null,
    traceId: 't1',
    ...overrides,
  };
}

describe('pass 21: an answer’s self-check', () => {
  it('keeps only the flagged sentences', () => {
    expect(
      flaggedFrom([
        { sentence: 'A [1].', status: 'supported' },
        { sentence: 'Staff get 30 days [1].', status: 'numbers_differ', missing_numbers: ['30'] },
        { sentence: 'Hi.', status: 'no_claim' },
        { sentence: 'The moon is cheese [1].', status: 'unsupported' },
      ]),
    ).toEqual([
      { sentence: 'Staff get 30 days [1].', status: 'numbers_differ', missingNumbers: ['30'] },
      { sentence: 'The moon is cheese [1].', status: 'unsupported' },
    ]);
    expect(flaggedFrom(undefined)).toEqual([]);
  });

  it('lists them under the answer, and keeps them through storage', () => {
    const turn = answer({
      flaggedSentences: [
        { sentence: 'Staff get 30 days [1].', status: 'numbers_differ', missingNumbers: ['30'] },
        { sentence: 'The moon is cheese [1].', status: 'unsupported' },
      ],
    });
    render(<ChatMessage turn={turn} engineerMode={false} />);
    const check = screen.getByTestId('answer-self-check');
    expect(check).toHaveTextContent('2 sentences aren’t backed by the passages they cite');
    expect(check).toHaveTextContent('the passages don’t state 30');
    expect(check).toHaveTextContent('no passage says this');
    expect(sanitizeTurn(turn)?.flaggedSentences).toEqual(turn.flaggedSentences);
  });

  it('shows nothing when every sentence is backed', () => {
    render(<ChatMessage turn={answer()} engineerMode={false} />);
    expect(screen.queryByTestId('answer-self-check')).toBeNull();
  });
});

describe('pass 21: flagged sentences marked in the answer', () => {
  it('marks plain text found once, and leaves anything else alone', async () => {
    const { markUnbacked } = await import('../../src/utils/citations');
    expect(markUnbacked('Staff get 30 days [1]. Pay is monthly [1].', ['Staff get 30 days [1].'])).toBe(
      '[Staff get 30 days](#docrag-unbacked) [1]. Pay is monthly [1].',
    );
    // Emphasis, too short, or not found as written: left as it was.
    expect(markUnbacked('**Staff** get 30 days [1].', ['**Staff** get 30 days [1].'])).toBe(
      '**Staff** get 30 days [1].',
    );
    expect(markUnbacked('Hi there [1].', ['Hi there [1].'])).toBe('Hi there [1].');
  });

  it('shows the marked sentence in the rendered answer', async () => {
    const { loadMarkdown } = await import('../../src/components/markdownLoader');
    await loadMarkdown();
    render(
      <ChatMessage
        turn={answer({
          content: 'Staff get 30 days of leave [1]. The moon is made of cheese [1].',
          sources: [
            { source_number: 1, filename: 'leave.txt', page: 1, section: 'Leave', chunk_id: 'c1', text: 'Staff get 28 days.' },
          ],
          flaggedSentences: [{ sentence: 'The moon is made of cheese [1].', status: 'unsupported' }],
        })}
        engineerMode={false}
      />,
    );
    expect(await screen.findByTestId('answer-unbacked')).toHaveTextContent('The moon is made of cheese');
    expect(screen.getAllByTestId('inline-citation')).toHaveLength(2);
  });
});

describe('pass 21: one verdict on an answer’s sentences', () => {
  it('lets the self-check stand in for the older per-passage counts', () => {
    render(<ChatMessage turn={answer({ unsupportedClaims: 2, selfChecked: true })} engineerMode={false} />);
    expect(screen.queryByTestId('chat-message-notes')).toBeNull();
  });

  it('keeps those counts when the server sent no self-check', () => {
    render(<ChatMessage turn={answer({ unsupportedClaims: 2 })} engineerMode={false} />);
    expect(screen.getByTestId('chat-message-notes')).toHaveTextContent('2 cited sentences match nothing');
  });

  it('keeps the self-check mark through storage', () => {
    expect(sanitizeTurn(answer({ selfChecked: true }))?.selfChecked).toBe(true);
  });
});
