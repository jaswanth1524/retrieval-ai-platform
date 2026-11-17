import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { ChatTurn } from '../../src/components/ChatMessage';
import ConversationList from '../../src/components/ConversationList';
import AnswerVersions from '../../src/components/chat/AnswerVersions';
import WhyThisAnswer from '../../src/components/chat/WhyThisAnswer';
import SavedPanel from '../../src/components/SavedPanel';
import FilePreview from '../../src/components/corpus/FilePreview';
import AdminTools from '../../src/components/settings/AdminTools';
import type { SavedAnswer } from '../../src/hooks/useChat';
import { COMPACT_ACTIONS_QUERY } from '../../src/hooks/useCompactActions';
import { BACKGROUND_IMPORT_STARTED_EVENT } from '../../src/utils/zipOptions';

const originalMatchMedia = window.matchMedia;
afterEach(() => {
  vi.restoreAllMocks();
  window.matchMedia = originalMatchMedia;
  localStorage.clear();
});

const answer = (id: string, extra: Partial<ChatTurn> = {}): ChatTurn => ({
  id,
  role: 'assistant',
  content: 'Ten days.',
  sources: [],
  timestamp: Date.now(),
  timings: null,
  traceId: null,
  ...extra,
});

describe('answer versions', () => {
  it('compares word by word while side by side, each column marking its own changes', async () => {
    render(
      <AnswerVersions
        versions={[{ content: 'Leave is ten days.', sources: [], timestamp: 1 }]}
        components={{}}
        current="Leave is twelve days."
      />,
    );
    await userEvent.click(screen.getByTestId('answer-versions-open'));
    await userEvent.click(screen.getByTestId('answer-version-side-by-side'));
    await userEvent.click(screen.getByTestId('answer-version-compare'));
    const before = screen.getByTestId('answer-version-diff-before');
    const after = screen.getByTestId('answer-version-diff-after');
    expect(before.querySelector('del')).toHaveTextContent('ten');
    expect(before.querySelector('ins')).toBeNull();
    expect(after.querySelector('ins')).toHaveTextContent('twelve');
    expect(after.querySelector('del')).toBeNull();
    // Back to one at a time, the comparison stays on.
    await userEvent.click(screen.getByTestId('answer-version-side-by-side'));
    expect(screen.getByTestId('answer-version-diff')).toBeInTheDocument();
  });
});

describe('why this answer', () => {
  it('says what was searched, how well it matched and what it cites', () => {
    render(
      <WhyThisAnswer
        turn={answer('a1', {
          searchedAs: 'How many days of annual leave?',
          topRerankScore: 0.2,
          rerankMinScore: 0.15,
          scoped: true,
          model: 'qwen2.5',
          sources: [
            { source_number: 1, filename: 'leave.md', page: 2, section: 'Annual', chunk_id: 'k1', text: '', rerank_score: 0.2 },
          ],
        })}
      />,
    );
    const why = screen.getByTestId('chat-message-why');
    expect(why).toHaveTextContent('searched as “How many days of annual leave?”');
    expect(why).toHaveTextContent('only in the documents selected');
    expect(screen.getByTestId('chat-message-why-score')).toHaveTextContent(
      'scored 0.20 out of 1 (passages under 0.15 aren’t used) — only just over the bar.',
    );
    expect(why).toHaveTextContent('It cites 1 passage from 1 document');
    expect(why).toHaveTextContent('[1] leave.md, p.2 · Annual · scored 0.20');
    expect(why).toHaveTextContent('Written by qwen2.5.');
  });

  it('a not-found answer says what was read', () => {
    render(
      <WhyThisAnswer
        turn={answer('a1', {
          notFound: true,
          contextSources: [
            { source_number: 1, filename: 'a.md', page: 1, section: 'S', chunk_id: 'k1', text: '' },
            { source_number: 2, filename: 'b.md', page: 1, section: 'S', chunk_id: 'k2', text: '' },
          ],
        })}
        weakSupportBelow={0.25}
      />,
    );
    expect(screen.getByTestId('chat-message-why')).toHaveTextContent(
      'The model read 2 passages and found no answer in them.',
    );
  });
});

describe('unsave', () => {
  it('moves focus to the next saved answer, then to the panel once none is left', async () => {
    const item = (id: string): SavedAnswer =>
      ({
        conversationId: 'c',
        conversationTitle: 'Leave',
        workspace: '',
        question: `Question ${id}`,
        turn: answer(id, { saved: true }),
      }) as SavedAnswer;
    function Harness() {
      const [items, setItems] = useState([item('a1'), item('a2')]);
      return (
        <SavedPanel
          answers={items}
          onOpen={() => {}}
          onUnsave={(_, turnId) => setItems((prev) => prev.filter((entry) => entry.turn.id !== turnId))}
        />
      );
    }
    render(<Harness />);
    await userEvent.click(screen.getAllByTestId('saved-unsave')[0]);
    expect(document.activeElement).toBe(screen.getByTestId('saved-open'));
    expect(document.activeElement).toHaveAccessibleName('Open the saved answer to “Question a2”');
    await userEvent.click(screen.getByTestId('saved-unsave'));
    expect(document.activeElement).toBe(screen.getByTestId('saved-empty'));
  });
});

describe('phone actions menu', () => {
  it('puts a conversation’s actions in one ⋯ menu on touch screens', async () => {
    window.matchMedia = ((query: string) => ({
      ...originalMatchMedia(query),
      matches: query === COMPACT_ACTIONS_QUERY,
      addEventListener: () => {},
      removeEventListener: () => {},
    })) as typeof window.matchMedia;
    const onDelete = vi.fn();
    const onPin = vi.fn();
    render(
      <ConversationList
        conversations={[{ id: 'c1', title: 'Leave', updatedAt: 1, turnCount: 2 }]}
        activeId="c1"
        onSwitch={vi.fn()}
        onRename={vi.fn()}
        onDelete={onDelete}
        onPin={onPin}
      />,
    );
    expect(screen.queryByTestId('conversation-pin')).toBeNull();
    const toggle = screen.getByTestId('conversation-menu');
    await userEvent.click(toggle);
    const menu = screen.getByRole('menu', { name: 'Actions for Leave' });
    expect(within(menu).getAllByRole('menuitem').map((item) => item.textContent)).toEqual([
      'Pin',
      'Rename',
      'Delete',
    ]);
    // Focus is in the menu; the arrows move through it and Escape goes back.
    expect(document.activeElement).toHaveTextContent('Pin');
    await userEvent.keyboard('{ArrowDown}');
    expect(document.activeElement).toHaveTextContent('Rename');
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('menu')).toBeNull();
    expect(document.activeElement).toBe(toggle);

    await userEvent.click(toggle);
    await userEvent.click(screen.getByTestId('conversation-menu-pin'));
    expect(onPin).toHaveBeenCalledWith('c1', true);
    await userEvent.click(toggle);
    await userEvent.click(screen.getByTestId('conversation-menu-delete'));
    // The usual confirm follows.
    expect(screen.getByTestId('conversation-delete-confirm')).toBeInTheDocument();
  });
});

describe('cancelling long admin requests', () => {
  it('a file preview can be cancelled, and its late result is not shown', async () => {
    let signal: AbortSignal | undefined;
    vi.spyOn(api, 'previewDocument').mockImplementation((_file, given) => {
      signal = given;
      return new Promise(() => {});
    });
    render(<FilePreview accept={['.md']} workspace="" />);
    const file = new File(['# A'], 'a.md', { type: 'text/markdown' });
    await userEvent.upload(screen.getByTestId('preview-file-input'), file);
    await userEvent.click(screen.getByTestId('preview-cancel'));
    expect(signal?.aborted).toBe(true);
    expect(screen.getByTestId('preview-file')).toHaveTextContent('Preview a file…');
    expect(screen.queryByTestId('preview-error')).toBeNull();
  });

  it('a stored backup check can be cancelled; a restore can run in the background', async () => {
    vi.spyOn(api, 'workspaceSettings').mockResolvedValue({ settings: {}, effective: {} } as never);
    vi.spyOn(api, 'answerCache').mockResolvedValue({ enabled: false, entries: [], total_entries: 0 } as never);
    vi.spyOn(api, 'listBackups').mockResolvedValue({
      backups: [{ name: 'docrag-20261101-000000.zip', created_at: 1, size: 10, verified: null }],
      interval_hours: 24,
      keep: 3,
    } as never);
    let signal: AbortSignal | undefined;
    vi.spyOn(api, 'verifyStoredBackup').mockImplementation((_name, given) => {
      signal = given;
      return new Promise(() => {});
    });
    const restore = vi.spyOn(api, 'restoreStoredBackup').mockResolvedValue({
      jobs: [],
      deferred: [],
      skipped: [],
      missing_originals: [],
      rejected: [],
      feedback_imported: 0,
      feedback_skipped: 0,
      import_id: 'imp1',
    });
    const started = vi.fn();
    window.addEventListener(BACKGROUND_IMPORT_STARTED_EVENT, started);
    try {
      render(<AdminTools />);
      await userEvent.click(await screen.findByTestId('backup-verify'));
      await userEvent.click(screen.getByTestId('backup-verify-cancel'));
      expect(signal?.aborted).toBe(true);
      expect(screen.getByText('Check cancelled.')).toBeInTheDocument();
      expect(screen.getByTestId('backup-verify')).toBeEnabled();

      await userEvent.click(screen.getByTestId('backups-restore-background'));
      await userEvent.click(screen.getByTestId('backup-restore'));
      await userEvent.click(screen.getByTestId('backup-restore-confirm'));
      expect(restore).toHaveBeenCalledWith('docrag-20261101-000000.zip', true);
      await waitFor(() => expect(screen.getByText(/Restoring docrag-20261101-000000.zip in the background/)).toBeInTheDocument());
      expect(started).toHaveBeenCalled();
    } finally {
      window.removeEventListener(BACKGROUND_IMPORT_STARTED_EVENT, started);
    }
  });
});
