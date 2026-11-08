import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';
import CitationCard from '../../src/components/CitationCard';
import ConversationList from '../../src/components/ConversationList';
import type { DocumentMeta } from '../../src/components/corpus/types';
import { coverageWorthShowing } from '../../src/hooks/chatPersistence';
import { withCurrentScope } from '../../src/hooks/useAskActions';
import { keepUnchanged } from '../../src/hooks/useCorpus';
import type { ConversationSummary } from '../../src/hooks/useChat';

const meta = (chunks: number, tags: string[] = []): DocumentMeta => ({
  chunks,
  stale: false,
  reindexable: false,
  tags,
});

const conversations = (count: number): ConversationSummary[] =>
  Array.from({ length: count }, (_, index) => ({
    id: `c${index}`,
    title: `Chat ${index}`,
    updatedAt: count - index,
    turnCount: 2,
  }));

const question = (content: string): ChatTurn => ({
  id: 'q1',
  role: 'user',
  content,
  sources: [],
  timestamp: Date.now(),
  timings: null,
  traceId: null,
});

describe('pass 15', () => {
  it('an unchanged listing keeps its objects, and a changed record only that one', () => {
    const prev = { 'a.md': meta(3, ['hr']), 'b.md': meta(5) };
    expect(keepUnchanged(prev, { 'a.md': meta(3, ['hr']), 'b.md': meta(5) })).toBe(prev);

    const next = keepUnchanged(prev, { 'a.md': meta(3, ['hr']), 'b.md': meta(6) });
    expect(next).not.toBe(prev);
    expect(next['a.md']).toBe(prev['a.md']);
    expect(next['b.md'].chunks).toBe(6);
    expect(keepUnchanged(prev, { 'a.md': meta(3, ['hr']) })).not.toBe(prev);
  });

  it('a question asked again takes the current scope page range, keeping its own narrowing', () => {
    const own = { page_min: 3, page_max: 5, exclude_chunk_ids: ['c1'], section: 'Leave' };
    expect(withCurrentScope(own, undefined)).toEqual({ exclude_chunk_ids: ['c1'], section: 'Leave' });
    expect(withCurrentScope(own, { page_min: 9 })).toEqual({
      exclude_chunk_ids: ['c1'],
      section: 'Leave',
      page_min: 9,
    });
    expect(withCurrentScope({ page_min: 3 }, undefined)).toBeUndefined();
    expect(withCurrentScope(undefined, { page_max: 2 })).toEqual({ page_max: 2 });
  });

  it('the conversation search stays while a search is typed, even below the threshold', async () => {
    const props = {
      activeId: 'c0',
      onSwitch: vi.fn(),
      onRename: vi.fn(),
      onDelete: vi.fn(),
    };
    const { rerender } = render(<ConversationList {...props} conversations={conversations(6)} />);
    await userEvent.type(screen.getByTestId('conversation-search'), 'Chat 1');

    rerender(<ConversationList {...props} conversations={conversations(5)} />);

    expect(screen.getByTestId('conversation-search')).toHaveValue('Chat 1');
    await userEvent.clear(screen.getByTestId('conversation-search'));
    expect(screen.queryByTestId('conversation-search')).toBeNull();
  });

  it('Enter in the edit box while an answer streams keeps the edit', async () => {
    const onEditQuestion = vi.fn();
    const turn = question('Old?');
    const { rerender } = render(
      <ChatMessage turn={turn} engineerMode={false} onEditQuestion={onEditQuestion} />,
    );
    fireEvent.click(screen.getByTestId('chat-message-edit'));
    // An answer starts (a follow-up chip, say) while the editor is open.
    rerender(<ChatMessage turn={turn} engineerMode={false} busy onEditQuestion={onEditQuestion} />);
    await userEvent.type(screen.getByTestId('chat-message-edit-input'), ' and new');
    await userEvent.keyboard('{Enter}');

    expect(onEditQuestion).not.toHaveBeenCalled();
    expect(screen.getByTestId('chat-message-edit-input')).toHaveValue('Old? and new');
    expect(screen.getByTestId('chat-message-edit-busy')).toBeInTheDocument();
  });

  it('the citation card tells a screen reader its score and that the document changed', () => {
    render(
      <CitationCard
        sourceNumber={2}
        filename="policy.pdf"
        page={4}
        section="Leave"
        score={0.734}
        change="changed"
      />,
    );
    const label = screen.getByTestId('citation-card').getAttribute('aria-label') ?? '';
    expect(label).toContain('Open policy.pdf at source 2');
    expect(label).toContain('match 0.73');
    expect(label).toMatch(/changed/);
  });

  it('a source marked not relevant keeps focus on its button', async () => {
    const onNotRelevant = vi.fn();
    const { rerender } = render(
      <CitationCard sourceNumber={1} filename="a.md" page={1} section="" onNotRelevant={onNotRelevant} />,
    );
    const button = screen.getByTestId('citation-not-relevant');
    button.focus();
    await userEvent.keyboard('{Enter}');
    expect(onNotRelevant).toHaveBeenCalledTimes(1);

    rerender(<CitationCard sourceNumber={1} filename="a.md" page={1} section="" markedNotRelevant />);
    expect(screen.getByTestId('citation-not-relevant')).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(onNotRelevant).toHaveBeenCalledTimes(1);
  });

  it('a summary that left a part out says so, even when it read every passage', () => {
    expect(coverageWorthShowing({ read: 8, total: 8 })).toBe(false);
    expect(coverageWorthShowing({ read: 8, total: 8, skipped: 1 })).toBe(true);
    expect(coverageWorthShowing({ read: 8, total: 8, cutOff: 2 })).toBe(true);
    render(
      <ChatMessage
        turn={{
          id: 'a1',
          role: 'assistant',
          content: 'Summary [1].',
          sources: [],
          timestamp: Date.now(),
          timings: null,
          traceId: null,
          coverage: { read: 8, total: 8, skipped: 1, cutOff: 2 },
        }}
        engineerMode={false}
      />,
    );
    expect(screen.getByTestId('chat-message-parts-skipped')).toHaveTextContent('1 part left out');
    expect(screen.getByTestId('chat-message-notes-cut')).toHaveTextContent('notes cut off on 2 parts');
    expect(screen.queryByTestId('chat-message-coverage')).toBeNull();
  });
});
