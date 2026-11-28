import { render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { ChatTurn } from '../../src/components/ChatMessage';
import SharedView from '../../src/components/SharedView';
import SharedLinks from '../../src/components/settings/SharedLinks';
import { useShares } from '../../src/hooks/useShares';
import {
  parseShareLink,
  sharedTurns,
  shareLinkUrl,
  turnsFromShare,
  withPassageText,
} from '../../src/utils/shares';

afterEach(() => {
  vi.restoreAllMocks();
  window.history.replaceState(null, '', '/');
});

const SOURCE = {
  source_number: 1,
  filename: 'travel.docx',
  page: 1,
  section: 'Meals',
  chunk_id: 'c1',
  text: 'The allowance is 60 euros in Paris.',
};

function turn(over: Partial<ChatTurn>): ChatTurn {
  return {
    id: 'x',
    role: 'assistant',
    content: '',
    sources: [],
    timestamp: 2000,
    timings: null,
    traceId: 'trace-1',
    ...over,
  };
}

const TURNS: ChatTurn[] = [
  turn({ id: 'q1', role: 'user', content: 'Meal allowance in Paris?' }),
  turn({
    id: 'a1',
    content: 'It is 60 euros [1].',
    sources: [SOURCE],
    model: 'gpt-5-mini',
    note: 'my private note',
    feedback: 'up',
  }),
  turn({ id: 'q2', role: 'user', content: 'And Berlin?' }),
  turn({ id: 'a2', content: 'Not in the documents.', notFound: true }),
  turn({ id: 'q3', role: 'user', content: 'Tokyo?' }),
  turn({ id: 'a3', content: 'It is', stopped: true }),
];

describe('pass 24: shared answers', () => {
  it('shares each answer with its question and sources, nothing private', () => {
    const shared = sharedTurns(TURNS);
    expect(shared).toEqual([
      {
        question: 'Meal allowance in Paris?',
        answer: 'It is 60 euros [1].',
        sources: [SOURCE],
        answer_status: 'answered',
        model: 'gpt-5-mini',
        asked_at: 2,
        truncated: false,
      },
      {
        question: 'And Berlin?',
        answer: 'Not in the documents.',
        sources: [],
        answer_status: 'insufficient_context',
        asked_at: 2,
        truncated: false,
      },
    ]);
    expect(JSON.stringify(shared)).not.toContain('private');
    expect(JSON.stringify(shared)).not.toContain('trace-1');
    expect(sharedTurns(TURNS, 'a2').map((t) => t.question)).toEqual(['And Berlin?']);
  });

  it('reads a share link and builds one', () => {
    expect(parseShareLink('#s=AbC_def-123456')).toBe('AbC_def-123456');
    expect(parseShareLink('#s=bad id')).toBeNull();
    expect(parseShareLink('#c=conversation')).toBeNull();
    expect(shareLinkUrl('AbC_def-123456', { origin: 'http://h', pathname: '/' })).toBe(
      'http://h/#s=AbC_def-123456',
    );
  });

  it('opens a share link from the address and takes it off', () => {
    window.history.replaceState(null, '', '/#s=AbC_def-123456');
    const push = vi.fn(() => 't');
    const { result } = renderHook(() =>
      useShares({ enabled: true, turns: [], conversationTitle: 'T', pushToast: push }),
    );
    expect(result.current.openShareId).toBe('AbC_def-123456');
    expect(window.location.hash).toBe('');
  });

  it('shares an answer and copies its link; off, it offers nothing', async () => {
    const create = vi.spyOn(api, 'createShare').mockResolvedValue({
      id: 'AbC_def-123456',
      title: 'Meal allowance in Paris?',
      workspace: 'default',
      created_at: 1,
      created_by: 'bob',
      turns: 1,
    });
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const push = vi.fn(() => 't');
    const { result } = renderHook(() =>
      useShares({ enabled: true, turns: TURNS, conversationTitle: 'Trip', pushToast: push }),
    );
    result.current.shareAnswer?.('a1');
    await waitFor(() => expect(push).toHaveBeenCalled());
    expect(create).toHaveBeenCalledWith({
      title: 'Meal allowance in Paris?',
      turns: sharedTurns(TURNS, 'a1'),
    });
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining('#s=AbC_def-123456'));
    expect(push).toHaveBeenCalledWith(expect.objectContaining({ title: 'Share link copied' }));
    const off = renderHook(() =>
      useShares({ enabled: false, turns: TURNS, conversationTitle: 'Trip', pushToast: push }),
    );
    expect(off.result.current.shareAnswer).toBeUndefined();
    expect(off.result.current.shareConversation).toBeUndefined();
  });

  it('shows a share read-only, with its passages, or why it cannot', async () => {
    vi.spyOn(api, 'readShare').mockResolvedValueOnce({
      id: 'AbC_def-123456',
      title: 'Trip',
      workspace: 'legal',
      created_at: 1,
      expires_at: null,
      created_by: 'bob',
      turns: 1,
      conversation: sharedTurns(TURNS, 'a1'),
    });
    const { unmount } = render(<SharedView shareId="AbC_def-123456" onClose={() => undefined} />);
    expect(await screen.findByText('Meal allowance in Paris?')).toBeTruthy();
    expect(screen.getByTestId('shared-view-meta').textContent).toContain('Shared by bob');
    expect(screen.getByTestId('shared-view-meta').textContent).toContain('from legal');
    expect(screen.getByText('The allowance is 60 euros in Paris.')).toBeTruthy();
    unmount();
    const { ApiClientError } = await import('../../src/api/errors');
    vi.spyOn(api, 'readShare').mockRejectedValueOnce(
      new ApiClientError('No such share (or it has expired).', 404),
    );
    render(<SharedView shareId="AbC_def-123456" onClose={() => undefined} />);
    expect((await screen.findByTestId('shared-view-error')).textContent).toContain('No such share');
  });

  it('lists this key’s shares and deletes one', async () => {
    vi.spyOn(api, 'listShares').mockResolvedValue({
      shares: [
        { id: 'AbC_def-123456', title: 'Trip', workspace: 'default', created_at: 1, created_by: 'bob', turns: 2 },
      ],
    });
    const remove = vi.spyOn(api, 'deleteShare').mockResolvedValue(undefined);
    render(<SharedLinks />);
    expect((await screen.findByTestId('shared-link')).textContent).toContain('Trip');
    expect(api.listShares).toHaveBeenCalledWith(true, expect.anything());
    await userEvent.click(screen.getByTestId('shared-link-delete'));
    await waitFor(() => expect(screen.queryByTestId('shared-link')).toBeNull());
    expect(remove).toHaveBeenCalledWith('AbC_def-123456');
  });

  it('fills in passage text older turns had dropped, and never fails over one', async () => {
    const shared = sharedTurns([
      turn({ id: 'q', role: 'user', content: 'Q' }),
      turn({
        id: 'a',
        content: 'A [1] [2] [3].',
        sources: [
          { ...SOURCE, text: '' },
          { ...SOURCE, source_number: 2, chunk_id: 'gone', text: '' },
          { ...SOURCE, source_number: 3, chunk_id: 'kept' },
        ],
      }),
    ]);
    const read = vi.fn(async (_filename: string, chunkId: string) => {
      if (chunkId === 'gone') throw new Error('404');
      return `text of ${chunkId}`;
    });
    const filled = await withPassageText(shared, read);
    expect(filled[0].sources.map((source) => source.text)).toEqual([
      'text of c1',
      '',
      'The allowance is 60 euros in Paris.',
    ]);
    expect(read).toHaveBeenCalledTimes(2);
  });

  it('imports a share as a conversation of this browser', async () => {
    const share = {
      id: 'AbC_def-123456',
      title: 'Trip',
      workspace: 'default',
      created_at: 1,
      expires_at: null,
      created_by: 'bob',
      turns: 2,
      conversation: sharedTurns(TURNS),
    };
    const turns = turnsFromShare(share);
    expect(turns.map((t) => [t.role, t.content, Boolean(t.notFound)])).toEqual([
      ['user', 'Meal allowance in Paris?', false],
      ['assistant', 'It is 60 euros [1].', false],
      ['user', 'And Berlin?', false],
      ['assistant', 'Not in the documents.', true],
    ]);
    vi.spyOn(api, 'readShare').mockResolvedValue(share);
    const onImport = vi.fn(() => true);
    const onClose = vi.fn();
    render(<SharedView shareId="AbC_def-123456" onClose={onClose} onImport={onImport} />);
    await userEvent.click(await screen.findByTestId('shared-view-import'));
    expect(onImport).toHaveBeenCalledWith(share);
    expect(onClose).toHaveBeenCalled();
  });
});
