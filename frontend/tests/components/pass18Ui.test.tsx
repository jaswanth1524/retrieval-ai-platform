import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { CitationResponse } from '../../src/api/types';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';
import ChatHeader from '../../src/components/ChatHeader';
import { loadMarkdown } from '../../src/components/markdownLoader';
import SavedPanel from '../../src/components/SavedPanel';
import AdminTools from '../../src/components/settings/AdminTools';
import { sanitizeTurn } from '../../src/hooks/chat/sanitize';
import type { SavedAnswer } from '../../src/hooks/useChat';
import { savedAnswersToMarkdown } from '../../src/utils/exportChat';
import { recallReading, rememberReading } from '../../src/utils/readingMemory';
import { readZipOptions, writeZipOptions } from '../../src/utils/zipOptions';

beforeAll(async () => {
  await loadMarkdown();
});

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

const source: CitationResponse = {
  source_number: 1,
  filename: 'leave.pdf',
  page: 2,
  section: 'Vacation',
  chunk_id: 'c1',
  text: 'Staff receive twenty-eight vacation days.',
} as CitationResponse;

function answer(overrides: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id: 'a1',
    role: 'assistant',
    content: 'Staff get 30 days [1].',
    sources: [source],
    timestamp: 1_700_000_000_000,
    timings: null,
    traceId: 't1',
    ...overrides,
  };
}

describe('pass 18 answers', () => {
  it('says what a follow-up was searched as and flags a number its sources lack', () => {
    render(
      <ChatMessage
        turn={answer({ searchedAs: 'How many vacation days do staff get?', unsupportedNumbers: 1 })}
        engineerMode={false}
      />,
    );
    expect(screen.getByTestId('chat-message-searched-as')).toHaveTextContent(
      'Searched as: How many vacation days do staff get?',
    );
    expect(screen.getByTestId('chat-message-notes')).toHaveTextContent(
      'One sentence states a number that the passages it cites don’t contain',
    );
  });

  it('offers to ask within the answer’s sources, and keeps notes and flags through storage', async () => {
    const onAskWithin = vi.fn();
    render(<ChatMessage turn={answer()} engineerMode={false} onAskWithin={onAskWithin} />);
    await userEvent.click(screen.getByTestId('chat-message-ask-within'));
    expect(onAskWithin).toHaveBeenCalledWith([source]);
    const kept = sanitizeTurn({
      ...answer(),
      note: '  check with HR  ',
      unsupportedClaims: 2,
      unsupportedNumbers: -1,
      searchedAs: 'x'.repeat(3000),
    });
    expect(kept?.note).toBe('  check with HR  ');
    expect(kept?.unsupportedClaims).toBe(2);
    expect(kept?.unsupportedNumbers).toBeUndefined();
    expect(kept?.searchedAs).toHaveLength(2000);
  });
});

describe('saved answers', () => {
  const item: SavedAnswer = {
    conversationId: 'conv',
    conversationTitle: 'Leave',
    workspace: 'hr',
    turn: answer({ note: 'Double-check' }),
    question: 'How many vacation days?',
  };

  it('takes a note and asks again', async () => {
    const onSetNote = vi.fn();
    const onAskAgain = vi.fn();
    const { rerender } = render(
      <SavedPanel
        answers={[{ ...item, turn: answer() }]}
        onOpen={vi.fn()}
        onUnsave={vi.fn()}
        onSetNote={onSetNote}
        onAskAgain={onAskAgain}
      />,
    );
    // Another workspace's answer can't be asked again from here (it would search these
    // documents).
    expect(screen.queryByTestId('saved-ask-again')).toBeNull();
    rerender(
      <SavedPanel
        answers={[{ ...item, turn: answer() }]}
        onOpen={vi.fn()}
        onUnsave={vi.fn()}
        onSetNote={onSetNote}
        onAskAgain={onAskAgain}
        workspace="hr"
      />,
    );
    await userEvent.click(screen.getByTestId('saved-note-edit'));
    await userEvent.type(screen.getByTestId('saved-note-input'), 'Ask HR');
    await userEvent.click(screen.getByTestId('saved-note-save'));
    expect(onSetNote).toHaveBeenCalledWith('conv', 'a1', 'Ask HR');
    await userEvent.click(screen.getByTestId('saved-ask-again'));
    expect(onAskAgain).toHaveBeenCalled();
  });

  it('exports notes and links back to each cited passage', () => {
    const markdown = savedAnswersToMarkdown([item], true);
    expect(markdown).toContain('**Note:** Double-check');
    expect(markdown).toMatch(/\[1\] \[leave\.pdf · p\.2 · Vacation\]\(http[^)]*#d=leave\.pdf&k=c1&w=hr\)/);
  });
});

describe('settings and options', () => {
  it('switches workspace from the chat header', async () => {
    const onSwitch = vi.fn();
    render(
      <ChatHeader
        title="Chat"
        scopeLabel="All documents"
        mode="reader"
        onSetMode={vi.fn()}
        hasTurns={false}
        onClear={vi.fn()}
        onExport={vi.fn()}
        onOpenPalette={vi.fn()}
        inspectorOpen={false}
        onToggleInspector={vi.fn()}
        workspace=""
        workspaces={['legal']}
        onSwitchWorkspace={onSwitch}
      />,
    );
    await userEvent.selectOptions(screen.getByTestId('chat-header-workspace'), 'legal');
    expect(onSwitch).toHaveBeenCalledWith('legal');
  });

  it('remembers zip options and the reading place per workspace', () => {
    expect(readZipOptions()).toEqual({ keepPaths: false, background: false });
    writeZipOptions({ keepPaths: true, background: true });
    expect(readZipOptions()).toEqual({ keepPaths: true, background: true });
    rememberReading('', 'a.pdf', 'c9');
    expect(recallReading('', 'a.pdf')).toBe('c9');
    expect(recallReading('legal', 'a.pdf')).toBeNull();
  });

  it('starts a rebuild, reads the audit log and tests notifications', async () => {
    vi.spyOn(api, 'workspaceSettings').mockResolvedValue({ settings: {}, effective: {} } as never);
    vi.spyOn(api, 'answerCache').mockResolvedValue({
      enabled: false,
      entries: [],
      total_entries: 0,
      max_entries: 0,
      ttl_seconds: 0,
      in_flight: 0,
    } as never);
    vi.spyOn(api, 'listBackups').mockRejectedValue(new Error('off'));
    vi.spyOn(api, 'rebuild').mockResolvedValue({ import_id: 'i1', originals: 3 });
    vi.spyOn(api, 'audit').mockResolvedValue({
      entries: [
        {
          created_at: Date.now() / 1000,
          workspace: 'default',
          event: 'document.delete',
          key: 'full',
          fields: { filename: 'a.pdf' },
        },
      ],
    });
    vi.spyOn(api, 'notifyTest').mockResolvedValue({ sent: false, error: 'hooks.example.com answered 403' });
    render(<AdminTools />);
    await userEvent.click(screen.getByTestId('rebuild-run'));
    await userEvent.click(screen.getByRole('button', { name: 'Rebuild' }));
    await waitFor(() => expect(screen.getByTestId('rebuild-result')).toHaveTextContent('Indexing 3 documents'));
    await userEvent.click(screen.getByTestId('audit-load'));
    expect(await screen.findByTestId('audit-entry')).toHaveTextContent('document.delete (full) filename=a.pdf');
    await userEvent.click(screen.getByTestId('notify-test-run'));
    expect(await screen.findByTestId('notify-test-result')).toHaveTextContent('Not sent: hooks.example.com answered 403');
  });
});
