import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import ReviewPanel from '../../src/components/ReviewPanel';

const answer = (id: string, extra: Partial<ChatTurn> = {}): ChatTurn => ({
  id,
  role: 'assistant',
  content: 'Ten days.',
  sources: [],
  timestamp: Date.now(),
  timings: null,
  traceId: null,
  uncited: true,
  ...extra,
});

function Harness() {
  const [turns, setTurns] = useState([answer('a1'), answer('a2')]);
  return (
    <ReviewPanel
      answers={turns.map((turn) => ({
        conversationId: 'c1',
        conversationTitle: 'Leave',
        workspace: '',
        question: `Question ${turn.id}`,
        turn,
      }))}
      onOpen={() => {}}
      onMarkReviewed={(_c, turnId, reviewed) =>
        setTurns((prev) =>
          prev.map((t) => (t.id === turnId ? { ...t, reviewedAt: reviewed ? 1 : undefined } : t)),
        )
      }
    />
  );
}

describe('pass 17 fixes', () => {
  it('moves focus to the next row when "Mark reviewed" removes one, then to the message', async () => {
    render(<Harness />);
    await userEvent.click(screen.getAllByTestId('review-mark')[0]);
    expect(screen.getAllByTestId('review-row')).toHaveLength(1);
    expect(document.activeElement).toBe(screen.getByTestId('review-open'));
    await userEvent.click(screen.getByTestId('review-mark'));
    expect(document.activeElement).toBe(screen.getByTestId('review-empty'));
  });
});

describe('stored turns and scopes', () => {
  it('fills a missing timing stage with 0 and bounds what a turn holds', async () => {
    const { sanitizeTurn, sanitizeScope } = await import('../../src/hooks/chat/sanitize');
    const turn = sanitizeTurn({
      id: 't',
      role: 'assistant',
      content: 'x'.repeat(300_000),
      sources: [],
      timestamp: 1,
      timings: { total_ms: 10 },
      traceId: 'y'.repeat(500),
      model: 'm'.repeat(1000),
    });
    expect(turn?.timings?.embed_ms).toBe(0);
    expect(turn?.timings?.total_ms).toBe(10);
    expect(turn?.content.length).toBe(200_000);
    expect(turn?.traceId).toBeNull();
    expect(turn?.model?.length).toBe(200);
    const scope = sanitizeScope({
      filenames: [...Array.from({ length: 600 }, (_, i) => `f${i}.md`), 'f1.md'],
      tags: ['a', 'a', ...Array.from({ length: 30 }, (_, i) => `t${i}`)],
    });
    expect(scope?.filenames).toHaveLength(500);
    expect(scope?.tags).toHaveLength(20);
    expect(new Set(scope?.tags).size).toBe(20);
  });
});

describe('cut-off answers', () => {
  it('offers Continue on a cut-off answer with a trace, and notes a cut prompt', async () => {
    const { default: ChatMessage } = await import('../../src/components/ChatMessage');
    const { vi } = await import('vitest');
    const onContinue = vi.fn();
    const { rerender } = render(
      <ChatMessage
        turn={answer('a1', { uncited: false, content: 'Alpha is', traceId: 't1', truncated: true, promptCut: true })}
        engineerMode={false}
        regenerateQuestion="What is alpha?"
        onRetry={vi.fn()}
        onContinue={onContinue}
      />,
    );
    await userEvent.click(screen.getByTestId('chat-message-continue'));
    expect(onContinue).toHaveBeenCalledWith('What is alpha?', undefined, 't1', 'a1');
    expect(screen.getByText(/read only part of what it was sent/)).toBeInTheDocument();

    rerender(
      <ChatMessage
        turn={answer('a1', { uncited: false, content: 'Alpha is', traceId: 't1' })}
        engineerMode={false}
        regenerateQuestion="What is alpha?"
        onRetry={vi.fn()}
        onContinue={onContinue}
      />,
    );
    expect(screen.queryByTestId('chat-message-continue')).not.toBeInTheDocument();
  });

  it('asks /questions/continue with the trace id when continuing', async () => {
    const { vi } = await import('vitest');
    const { api } = await import('../../src/api/client');
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('data: {"type":"done","answer":"Alpha is beta.","sources":[],"timings":{"embed_ms":0,"search_ms":0,"rerank_ms":0,"generate_ms":1,"total_ms":1,"condense_ms":0},"trace_id":"t2"}\n\n', {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      }),
    );
    const done = vi.fn();
    await api.askQuestionStream('What is alpha?', undefined, undefined, undefined, undefined, { onSources: () => {}, onDelta: () => {}, onDone: done } as never, undefined, { continueFrom: 't1' });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/questions/continue');
    expect(JSON.parse(String((init as RequestInit).body))).toEqual({ trace_id: 't1' });
    fetchMock.mockRestore();
  });
});

describe('score report from ratings', () => {
  it('asks for the ratings-based report when the box is ticked', async () => {
    const { vi } = await import('vitest');
    const { ScoreReportView } = await import('../../src/components/TraceTools');
    const report = {
      current_min_score: 0.15,
      bin_edges: [0, 0.5, 1],
      answered: [0, 1],
      unanswered: [1, 0],
      answered_count: 1,
      unanswered_count: 1,
      note: 'n',
    };
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () => new Response(JSON.stringify(report), { status: 200, headers: { 'Content-Type': 'application/json' } }),
    );
    render(<ScoreReportView onClose={() => {}} ratingsAvailable />);
    await screen.findByText(/1 answered/);
    await userEvent.click(screen.getByTestId('score-report-by-ratings'));
    await screen.findByText(/1 answered/);
    const urls = fetchMock.mock.calls.map((call) => String(call[0]));
    expect(urls.some((url) => url.includes('labels=feedback'))).toBe(true);
    fetchMock.mockRestore();
  });
});

describe('recently deleted', () => {
  it('keeps a deleted conversation for a while and gives it back', async () => {
    const { addToTrash, readTrash } = await import('../../src/utils/trash');
    const { default: RecentlyDeleted } = await import('../../src/components/RecentlyDeleted');
    const { vi } = await import('vitest');
    localStorage.clear();
    const conversation = {
      id: 'c1',
      title: 'Leave questions',
      turns: [answer('t1', { uncited: false })],
      createdAt: 1,
      updatedAt: 2,
    };
    addToTrash(conversation as never, Date.now());
    // Older than the tombstone window: gone.
    addToTrash({ ...conversation, id: 'old', title: 'Old' } as never, Date.now() - 40 * 86_400_000);
    expect(readTrash().map((entry) => entry.conversation.id)).toEqual(['c1']);

    const onRestore = vi.fn();
    render(<RecentlyDeleted version={0} onRestore={onRestore} />);
    await userEvent.click(screen.getByText(/Recently deleted \(1\)/));
    await userEvent.click(screen.getByTestId('recently-deleted-restore'));
    expect(onRestore).toHaveBeenCalledWith(expect.objectContaining({ id: 'c1' }));
    expect(readTrash()).toEqual([]);
  });
});

describe('viewer contents', () => {
  it('lists sections and goes to the page asked for', async () => {
    const { vi } = await import('vitest');
    const { api } = await import('../../src/api/client');
    const { default: ViewerOutline } = await import('../../src/components/ViewerOutline');
    vi.spyOn(api, 'documentOutline').mockResolvedValue({
      filename: 'h.pdf',
      total_chunks: 9,
      sections: [
        { section: 'Leave', page: 1, chunk_id: 'c1' },
        { section: 'Travel', page: 4, chunk_id: 'c5' },
      ],
      pages: [
        { section: 'Leave', page: 1, chunk_id: 'c1' },
        { section: 'Leave', page: 2, chunk_id: 'c3' },
        { section: 'Travel', page: 4, chunk_id: 'c5' },
      ],
    });
    const onGoTo = vi.fn();
    render(<ViewerOutline filename="h.pdf" onGoTo={onGoTo} />);
    await userEvent.click(await screen.findByText('Travel'));
    expect(onGoTo).toHaveBeenLastCalledWith('c5');
    // Page 3 has no passage of its own: the one running onto it (page 2's).
    await userEvent.type(screen.getByTestId('viewer-goto-page'), '3');
    await userEvent.click(screen.getByTestId('viewer-goto-submit'));
    expect(onGoTo).toHaveBeenLastCalledWith('c3');
    vi.restoreAllMocks();
  });
});

describe('saved answers check', () => {
  it('flags a saved answer whose passages no longer rank', async () => {
    const { vi } = await import('vitest');
    const { api } = await import('../../src/api/client');
    const { default: SavedPanel } = await import('../../src/components/SavedPanel');
    const source = (chunk_id: string) => ({
      source_number: 1,
      filename: 'a.md',
      page: 1,
      section: 'S',
      chunk_id,
      text: '',
    });
    const saved = (id: string, chunk: string) => ({
      conversationId: 'c',
      conversationTitle: 'Leave',
      workspace: '',
      question: `Question ${id}`,
      turn: answer(id, { uncited: false, saved: true, sources: [source(chunk)] }),
    });
    const search = vi
      .spyOn(api, 'searchPassages')
      .mockResolvedValueOnce({ results: [{ ...source('k1'), rank: 1 }], timings: {} } as never)
      .mockResolvedValueOnce({ results: [{ ...source('other'), rank: 1 }], timings: {} } as never);
    render(
      <SavedPanel answers={[saved('a1', 'k1'), saved('a2', 'k2')]} onOpen={() => {}} onUnsave={() => {}} />,
    );
    await userEvent.click(screen.getByTestId('saved-check'));
    const results = await screen.findAllByTestId('saved-check-result');
    await screen.findByText(/no longer rank/);
    expect(results[0]).toHaveTextContent('Still found');
    expect(search).toHaveBeenCalledTimes(2);
    vi.restoreAllMocks();
  });
});

describe('asking another model', () => {
  it('offers the other usable providers in Reformat and passes the model on', async () => {
    const { vi } = await import('vitest');
    const { default: ReformatMenu } = await import('../../src/components/chat/ReformatMenu');
    const { otherUsableProviders } = await import('../../src/utils/providers');
    const config = {
      ollama_available: true,
      openai_available: true,
      llm_model: 'qwen2.5',
      openai_model: 'gpt-4o-mini',
    } as never;
    const others = otherUsableProviders('ollama', config);
    expect(others).toEqual([{ id: 'openai', label: 'gpt-4o-mini' }]);
    const onReformat = vi.fn();
    render(<ReformatMenu onReformat={onReformat} providers={others} />);
    await userEvent.click(screen.getByTestId('chat-message-reformat'));
    await userEvent.click(screen.getByTestId('chat-message-ask-openai'));
    expect(onReformat).toHaveBeenCalledWith({ provider: 'openai', modelLabel: 'gpt-4o-mini' });
  });

  it('shows an earlier version next to the current answer', async () => {
    const { default: AnswerVersions } = await import('../../src/components/chat/AnswerVersions');
    render(
      <AnswerVersions
        versions={[{ content: 'Old answer.', sources: [], timestamp: 1, model: 'qwen2.5' }]}
        components={{}}
        current="New answer."
      />,
    );
    await userEvent.click(screen.getByTestId('answer-versions-open'));
    await userEvent.click(screen.getByTestId('answer-version-side-by-side'));
    const side = screen.getByTestId('answer-version-side');
    expect(side).toHaveTextContent('This version · qwen2.5');
    expect(side).toHaveTextContent('The current answer');
  });
});

describe('selection actions', () => {
  it('offers ask, search and copy for text selected in a passage', async () => {
    const { vi } = await import('vitest');
    const { act } = await import('@testing-library/react');
    const { default: SelectionActions } = await import('../../src/components/SelectionActions');
    const onAsk = vi.fn();
    const onSearch = vi.fn();
    const passage = document.createElement('div');
    passage.setAttribute('data-quote-source', 'policy.pdf, p. 3');
    passage.textContent = 'Employees accrue twenty days of leave.';
    document.body.appendChild(passage);
    Range.prototype.getBoundingClientRect = () =>
      ({ top: 100, left: 20, width: 50, height: 10, bottom: 110, right: 70 }) as DOMRect;
    render(<SelectionActions onAsk={onAsk} onSearch={onSearch} />);
    await act(async () => {
      const range = document.createRange();
      range.selectNodeContents(passage);
      const selection = document.getSelection()!;
      selection.removeAllRanges();
      selection.addRange(range);
      document.dispatchEvent(new Event('selectionchange'));
      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    });
    await userEvent.click(await screen.findByTestId('selection-search'));
    expect(onSearch).toHaveBeenCalledWith('Employees accrue twenty days of leave.');
    passage.remove();
  });
});
