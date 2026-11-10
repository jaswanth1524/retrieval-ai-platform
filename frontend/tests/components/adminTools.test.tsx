import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError, api } from '../../src/api/client';
import AdminTools from '../../src/components/settings/AdminTools';

afterEach(() => vi.restoreAllMocks());

function serve(): void {
  vi.spyOn(api, 'workspaceSettings').mockResolvedValue({
    settings: {},
    effective: { rerank_min_score: 0.15 },
  } as never);
  vi.spyOn(api, 'putWorkspaceSettings').mockResolvedValue({
    settings: { rerank_min_score: 0.3 },
    effective: { rerank_min_score: 0.3 },
  } as never);
  vi.spyOn(api, 'listBackups').mockResolvedValue({
    backups: [{ name: 'docrag-20261101-000000.zip', created_at: 1, size: 10, verified: true }],
    interval_hours: 24,
    keep: 3,
  } as never);
}

const cache = (entries: number) =>
  ({
    enabled: true,
    entries: Array.from({ length: entries }, (_, n) => ({ question: `q${n}`, age_seconds: n })),
    total_entries: entries,
    max_entries: 100,
    ttl_seconds: 600,
    in_flight: 0,
  }) as never;

describe('AdminTools', () => {
  it('a stored backup restored hands its jobs on, to be followed', async () => {
    serve();
    vi.spyOn(api, 'answerCache').mockResolvedValue(cache(0));
    vi.spyOn(api, 'restoreStoredBackup').mockResolvedValue({
      jobs: [{ job_id: 'j1', filename: 'a.md' }],
      skipped: [],
    } as never);
    const onRestored = vi.fn();
    render(<AdminTools onRestored={onRestored} />);
    await userEvent.click(await screen.findByTestId('backup-restore'));
    await userEvent.click(screen.getByTestId('backup-restore-confirm'));
    await waitFor(() => expect(onRestored).toHaveBeenCalledWith([{ job_id: 'j1', filename: 'a.md' }]));
  });

  it('saving the workspace settings reads the cache again, and a good read clears an error', async () => {
    serve();
    const read = vi
      .spyOn(api, 'answerCache')
      .mockRejectedValueOnce(new ApiClientError('down', 503))
      .mockResolvedValue(cache(0));
    render(<AdminTools />);
    expect(await screen.findByText(/down/)).toBeInTheDocument();
    await userEvent.click(await screen.findByTestId('workspace-settings-save'));
    await waitFor(() => expect(read).toHaveBeenCalledTimes(2));
    expect(await screen.findByTestId('answer-cache-count')).toHaveTextContent('0 in this workspace');
    expect(screen.queryByText(/down/)).not.toBeInTheDocument();
  });
});

describe('pass 16 focus', () => {
  it('Reformat moves focus into the menu and back, and Escape closes it', async () => {
    const { default: ReformatMenu } = await import('../../src/components/chat/ReformatMenu');
    const onReformat = vi.fn();
    render(<ReformatMenu onReformat={onReformat} />);
    await userEvent.click(screen.getByTestId('chat-message-reformat'));
    expect(document.activeElement).toBe(screen.getByTestId('chat-message-reformat-bullets'));
    await userEvent.keyboard('{Escape}');
    expect(document.activeElement).toBe(screen.getByTestId('chat-message-reformat'));
    await userEvent.click(screen.getByTestId('chat-message-reformat'));
    await userEvent.type(screen.getByTestId('chat-message-reformat-language'), 'pt-BR{Enter}');
    expect(screen.getByRole('alert')).toHaveTextContent('language name in letters');
    expect(onReformat).not.toHaveBeenCalled();
    await userEvent.click(screen.getByTestId('chat-message-reformat-table'));
    expect(onReformat).toHaveBeenCalledWith({ answerFormat: 'table' });
    expect(document.activeElement).toBe(screen.getByTestId('chat-message-reformat'));
  });

  it('earlier answers name their style, and hiding them returns focus to the toggle', async () => {
    const { default: AnswerVersions } = await import('../../src/components/chat/AnswerVersions');
    render(
      <AnswerVersions
        versions={[{ content: 'Old.', sources: [], timestamp: 1, style: { answerFormat: 'table', answerLanguage: 'German' } }]}
        components={{}}
      />,
    );
    await userEvent.click(screen.getByTestId('answer-versions-open'));
    expect(screen.getByTestId('answer-version-meta')).toHaveTextContent('table, German');
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Hide earlier answers' }));
    await userEvent.click(screen.getByRole('button', { name: 'Hide earlier answers' }));
    expect(document.activeElement).toBe(screen.getByTestId('answer-versions-open'));
  });
});

describe('pass 16 admin features', () => {
  it('lists cached questions, and a checked upload says what a restore would do', async () => {
    serve();
    vi.spyOn(api, 'answerCache').mockResolvedValue({
      enabled: true,
      entries: [{ question: 'What is leave?', age_seconds: 90 }],
      total_entries: 1,
      max_entries: 100,
      ttl_seconds: 600,
      in_flight: 1,
    } as never);
    vi.spyOn(api, 'verifyBackup').mockResolvedValue({
      ok: true,
      documents: 3,
      originals: 3,
      originals_missing: 0,
      versions: 0,
      feedback: false,
      problems: [],
      would_restore: 2,
      rejected: ['big.pdf: larger than MAX_UPLOAD_BYTES'],
    });
    render(<AdminTools />);
    expect(await screen.findByTestId('answer-cache-entry')).toHaveTextContent('What is leave? · 2 min ago');
    expect(screen.getByTestId('answer-cache-count')).toHaveTextContent('1 being answered now');
    await userEvent.upload(screen.getByTestId('backup-check-file'), new File(['PK'], 'b.zip'));
    expect(await screen.findByTestId('backup-check-result')).toHaveTextContent(
      'b.zip: 3 document(s), 2 a restore would bring back — a restore would refuse 1: big.pdf: larger than MAX_UPLOAD_BYTES',
    );
  });

  it('repairs only what is ticked, after a confirm', async () => {
    serve();
    vi.spyOn(api, 'answerCache').mockResolvedValue(cache(0));
    const report = {
      documents: 2,
      originals_kept: true,
      originals_without_document: ['orphan.txt'],
      documents_without_original: [],
      versions_without_document: [],
      incomplete: [],
      stale: [],
      duplicate_groups: [],
      in_flight: [],
      problems: 1,
    };
    vi.spyOn(api, 'consistency').mockResolvedValue(report as never);
    const repaired = vi.spyOn(api, 'repairConsistency').mockResolvedValue({
      originals_deleted: ['orphan.txt'],
      versions_deleted: [],
      reindex_jobs: [],
      left: [],
    });
    render(<AdminTools />);
    await userEvent.click(screen.getByTestId('consistency-run'));
    expect(screen.queryByTestId('consistency-repair-reindex_incomplete')).not.toBeInTheDocument();
    await userEvent.click(await screen.findByTestId('consistency-repair-delete_orphan_originals'));
    await userEvent.click(screen.getByTestId('consistency-repair-run'));
    await userEvent.click(screen.getByTestId('consistency-repair-confirm'));
    await waitFor(() => expect(repaired).toHaveBeenCalledWith({ delete_orphan_originals: true }));
    expect(await screen.findByTestId('consistency-repaired')).toHaveTextContent('1 put right.');
  });
});

describe('sitemap imports', () => {
  it('show how far an import is and cancel it', async () => {
    const { default: SitemapImports } = await import('../../src/components/corpus/SitemapImports');
    const running = {
      id: 'i1',
      url: 'https://example.com/sitemap.xml',
      state: 'running',
      total: 10,
      done: 4,
      queued: 3,
      failed: 1,
      skipped: 0,
      unchanged: 0,
      cancelled_pages: 0,
      started_at: Date.now() / 1000,
      finished_at: null,
      problems: ['https://example.com/x: The URL answered 404.'],
    };
    vi.spyOn(api, 'listSitemapImports').mockResolvedValue({ imports: [running] });
    const cancel = vi
      .spyOn(api, 'cancelSitemapImport')
      .mockResolvedValue({ ...running, state: 'cancelling' });
    render(<SitemapImports workspace="" canCancel />);
    const line = await screen.findByTestId('sitemap-import');
    expect(line).toHaveTextContent('example.com: 4 of 10 page(s) — 3 added, 0 unchanged, 1 failed');
    expect(line).toHaveTextContent('answered 404');
    await userEvent.click(screen.getByTestId('sitemap-import-cancel'));
    expect(cancel).toHaveBeenCalledWith('i1', '');
    expect(await screen.findByText(/stopping/)).toBeInTheDocument();
  });
});

describe('answer versions compared and used', () => {
  it('compares a version with the current answer and makes it the answer', async () => {
    const { default: AnswerVersions } = await import('../../src/components/chat/AnswerVersions');
    const onUse = vi.fn();
    render(
      <AnswerVersions
        versions={[{ content: 'Leave is ten days.', sources: [], timestamp: 1 }]}
        current="Leave is twelve days."
        onUse={onUse}
        components={{}}
      />,
    );
    await userEvent.click(screen.getByTestId('answer-versions-open'));
    await userEvent.click(screen.getByTestId('answer-version-compare'));
    const diff = screen.getByTestId('answer-version-diff');
    expect(diff.querySelector('del')).toHaveTextContent('ten');
    expect(diff.querySelector('ins')).toHaveTextContent('twelve');
    await userEvent.click(screen.getByTestId('answer-version-use'));
    expect(onUse).toHaveBeenCalledWith(0);
  });
});

describe('needs review', () => {
  it('marks answers reviewed, hides them, and counts the rest on the rail', async () => {
    const { default: ReviewPanel } = await import('../../src/components/ReviewPanel');
    const { default: IconRail } = await import('../../src/components/IconRail');
    const { unreviewedCount } = await import('../../src/utils/review');
    const flagged = (id: string, reviewedAt?: number) => ({
      conversationId: 'c',
      conversationTitle: 'Plans',
      workspace: '',
      question: `Question ${id}?`,
      turn: {
        id,
        role: 'assistant' as const,
        content: 'Not in the documents.',
        sources: [],
        timestamp: Date.now(),
        timings: null,
        traceId: null,
        notFound: true,
        ...(reviewedAt ? { reviewedAt } : {}),
      },
    });
    const onMark = vi.fn();
    const answers = [flagged('a1'), flagged('a2', 5)];
    render(<ReviewPanel answers={answers} onOpen={() => {}} onMarkReviewed={onMark} />);
    expect(screen.getAllByTestId('review-row')).toHaveLength(1);
    await userEvent.click(screen.getByTestId('review-mark'));
    expect(onMark).toHaveBeenCalledWith('c', 'a1', true);
    await userEvent.click(screen.getByTestId('review-show-reviewed'));
    expect(screen.getAllByTestId('review-row')).toHaveLength(2);

    expect(unreviewedCount(answers.map((item) => item.turn))).toBe(1);
    render(
      <IconRail
        active="chat"
        onSelect={() => {}}
        theme="light"
        onToggleTheme={() => {}}
        onOpenSettings={() => {}}
        badges={{ review: 1 }}
      />,
    );
    expect(screen.getByTestId('rail-badge-review')).toHaveTextContent('1');
  });
});

describe('label management', () => {
  it('renames, removes and exports a label from its filter', async () => {
    const { default: ConversationList } = await import('../../src/components/ConversationList');
    const onRenameLabel = vi.fn();
    const onExportLabel = vi.fn();
    render(
      <ConversationList
        conversations={[{ id: 'c1', title: 'Leave', updatedAt: 2, turnCount: 2, labels: ['hr'] }]}
        activeId={null}
        onSwitch={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        onSetLabels={vi.fn()}
        onRenameLabel={onRenameLabel}
        onExportLabel={onExportLabel}
      />,
    );
    expect(screen.queryByTestId('conversation-label-tools')).not.toBeInTheDocument();
    await userEvent.click(screen.getByTestId('conversation-label-filter'));
    await userEvent.click(screen.getByTestId('conversation-label-export'));
    expect(onExportLabel).toHaveBeenCalledWith('hr');
    await userEvent.click(screen.getByTestId('conversation-label-remove'));
    await userEvent.click(screen.getByTestId('conversation-label-remove-confirm'));
    expect(onRenameLabel).toHaveBeenLastCalledWith('hr', null);
    await userEvent.click(screen.getByTestId('conversation-label-filter'));
    await userEvent.click(screen.getByTestId('conversation-label-rename'));
    const input = screen.getByTestId('conversation-label-rename-input');
    await userEvent.clear(input);
    await userEvent.type(input, 'people{Enter}');
    expect(onRenameLabel).toHaveBeenLastCalledWith('hr', 'people');
  });

  it('renaming a label reaches every conversation that has it', async () => {
    const { renderHook, act } = await import('@testing-library/react');
    const { useChat } = await import('../../src/hooks/useChat');
    localStorage.clear();
    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.historyReady).toBe(true));
    const first = result.current.activeConversationId as string;
    act(() => result.current.setLabels(first, ['HR', 'ops']));
    act(() => result.current.renameLabel('hr', 'people'));
    const labels = () =>
      result.current.conversations.find((conversation) => conversation.id === first)?.labels;
    expect(labels()).toEqual(['people', 'ops']);
    act(() => result.current.renameLabel('ops', null));
    expect(labels()).toEqual(['people']);
    expect(JSON.parse(result.current.allConversationsJson('people')).docrag_conversations).toEqual([]);
  });
});

describe('regenerated earlier answer', () => {
  it('the view follows it to its place', async () => {
    const { default: ChatThread } = await import('../../src/components/ChatThread');
    const t = (id: string, role: 'user' | 'assistant', extra = {}) => ({
      id,
      role,
      content: id,
      sources: [],
      timestamp: 1,
      timings: null,
      traceId: null,
      ...extra,
    });
    const scrolled: string[] = [];
    const original = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = function (this: Element) {
      const id = (this as HTMLElement).dataset?.turnId;
      if (id) scrolled.push(id);
    };
    try {
      const before = [t('q1', 'user'), t('a1', 'assistant'), t('q2', 'user'), t('a2', 'assistant')];
      const { rerender } = render(<ChatThread turns={before} pending={false} engineerMode={false} />);
      expect(scrolled).toEqual([]);
      const after = [t('q3', 'user'), t('a3', 'assistant', { previousIds: ['a1'] }), t('q2', 'user'), t('a2', 'assistant')];
      rerender(<ChatThread turns={after} pending={false} engineerMode={false} />);
      expect(scrolled).toContain('a3');
    } finally {
      Element.prototype.scrollIntoView = original;
    }
  });
});
