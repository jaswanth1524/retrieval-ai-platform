import { createEvent, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { RecheckResponse } from '../../src/api/types';
import Composer from '../../src/components/Composer';
import ConversationList from '../../src/components/ConversationList';
import DocumentViewer from '../../src/components/DocumentViewer';
import SavedPanel from '../../src/components/SavedPanel';
import type { ConversationSummary } from '../../src/hooks/useChat';
import { MAX_QUESTION_CHARS } from '../../src/utils/limits';
import { saveTemplate } from '../../src/utils/questionTemplates';
import { noLongerHolds, recheckSummary } from '../../src/utils/savedCheck';

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

function composer() {
  return (
    <Composer
      onSubmit={vi.fn()}
      disabled={false}
      pending={false}
      onCancel={vi.fn()}
      scopeLabel="All documents"
      indexedFilenames={[]}
      selectedFilenames={[]}
      onSelectedFilenamesChange={vi.fn()}
      config={null}
      provider="ollama"
      onProviderChange={vi.fn()}
      providerLabel="model"
    />
  );
}

const box = () => screen.getByTestId<HTMLTextAreaElement>('question-textarea');

describe('pass 23: the question box', () => {
  it('leaves Tab alone in a question that only happens to hold braces', () => {
    render(composer());
    fireEvent.change(box(), { target: { value: 'What is \\frac{a}{b}?' } });
    box().setSelectionRange(0, 0);
    const event = createEvent.keyDown(box(), { key: 'Tab' });
    fireEvent(box(), event);
    expect(event.defaultPrevented).toBe(false);
  });

  it('cuts a template added to a long draft at the question limit', async () => {
    saveTemplate('Long', 'x'.repeat(2000));
    render(composer());
    fireEvent.change(box(), { target: { value: 'y'.repeat(MAX_QUESTION_CHARS - 100) } });
    await userEvent.click(screen.getByTestId('composer-templates-button'));
    await userEvent.click(screen.getByTestId('template-insert'));
    expect(box().value).toHaveLength(MAX_QUESTION_CHARS);
  });
});

describe('pass 23: the viewer', () => {
  it('keeps Alt+arrows from the browser with no passage to step to', () => {
    vi.spyOn(api, 'getDocumentContent').mockReturnValue(new Promise(() => {}));
    render(<DocumentViewer filename="guide.md" chunkId="c1" onClose={vi.fn()} docked />);
    const target = screen.getByTestId('document-viewer');
    const event = createEvent.keyDown(target, { key: 'ArrowLeft', altKey: true });
    fireEvent(target, event);
    expect(event.defaultPrevented).toBe(true);
  });
});

function conversations(count: number, pinned = 0): ConversationSummary[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `c${index}`,
    title: `Chat ${index}`,
    updatedAt: count - index,
    turnCount: 1,
    ...(index < pinned ? { pinned: true } : {}),
  }));
}

describe('pass 23: selecting conversations', () => {
  const list = (items: ConversationSummary[], onPin = vi.fn()) => (
    <ConversationList
      conversations={items}
      activeId="c0"
      onSwitch={vi.fn()}
      onRename={vi.fn()}
      onDelete={vi.fn()}
      onDeleteMany={vi.fn()}
      onPin={onPin}
    />
  );

  it('ends when only one conversation is left', async () => {
    const { rerender } = render(list(conversations(2)));
    await userEvent.click(screen.getByTestId('conversation-select-mode'));
    expect(screen.getAllByTestId('conversation-pick')).toHaveLength(2);
    rerender(list(conversations(1)));
    expect(screen.queryByTestId('conversation-pick')).toBeNull();
  });

  it('says when only some of those picked could be pinned', async () => {
    const onPin = vi.fn();
    render(list(conversations(13, 8), onPin));
    await userEvent.click(screen.getByTestId('conversation-select-mode'));
    await userEvent.click(screen.getByTestId('conversation-select-all'));
    await userEvent.click(screen.getByTestId('conversation-bulk-pin'));
    expect(onPin).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId('conversation-pin-note')).toHaveTextContent('Pinned 2 of 5');
  });
});

const LONG: RecheckResponse = {
  sources: [{ source_number: 1, filename: 'a.txt', chunk_id: 'c1', state: 'changed' }],
  sentences: [{ sentence: 'x', status: 'supported' }],
  still_holds: false,
  more: true,
};

describe('pass 23: saved answers', () => {
  it('don’t read a long answer that held as far as it was read as failing', () => {
    expect(noLongerHolds(LONG)).toBe(false);
    expect(recheckSummary(LONG)).toMatch(/^Holds so far/);
  });

  it('show what the watch found on the answer it judged, and no other', () => {
    const turn = {
      id: 't',
      role: 'assistant' as const,
      content: 'Staff get 28 days [1].',
      sources: [],
      timestamp: 1,
      timings: null,
      traceId: null,
      saved: true,
    };
    const item = { conversationId: 'c', conversationTitle: 'T', workspace: '', turn, question: 'Q?' };
    const panel = (answer: string) => (
      <SavedPanel
        answers={[item]}
        onOpen={vi.fn()}
        onUnsave={vi.fn()}
        verdicts={{ 'c:t': { answer, text: 'Check it: 1 sentence no longer backed.' } }}
      />
    );
    const { rerender } = render(panel(turn.content));
    expect(screen.getByTestId('saved-watch-result')).toHaveTextContent('no longer backed');
    rerender(panel('An earlier answer.'));
    expect(screen.queryByTestId('saved-watch-result')).toBeNull();
  });
});
