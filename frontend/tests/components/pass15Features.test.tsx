import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';
import ConversationList from '../../src/components/ConversationList';
import ReviewPanel from '../../src/components/ReviewPanel';
import SummaryDialog from '../../src/components/SummaryDialog';
import ToastRow from '../../src/components/ToastRow';
import DocumentCard from '../../src/components/corpus/DocumentCard';
import { loadMarkdown } from '../../src/components/markdownLoader';

const answer = (extra: Partial<ChatTurn> = {}): ChatTurn => ({
  id: 'a1',
  role: 'assistant',
  content: 'Ten days [1].',
  sources: [{ source_number: 1, filename: 'a.md', page: 1, section: 'S', chunk_id: 'c1', text: '' }],
  timestamp: Date.now(),
  timings: null,
  traceId: null,
  ...extra,
});

describe('pass 15 features', () => {
  it('a toast offers its action and goes once it is used', async () => {
    const run = vi.fn();
    const onDismiss = vi.fn();
    render(
      <ToastRow
        toasts={[{ id: 't', title: 'Deleted “Plans”', tone: 'info', action: { label: 'Undo', run } }]}
        onDismiss={onDismiss}
      />,
    );
    await userEvent.click(screen.getByTestId('toast-action'));
    expect(run).toHaveBeenCalledOnce();
    expect(onDismiss).toHaveBeenCalledWith('t');
  });

  it('earlier answers are stepped through under the current one', async () => {
    await loadMarkdown();
    render(
      <ChatMessage
        turn={answer({
          versions: [
            { content: 'Older one.', sources: [], timestamp: 1 },
            { content: 'Newer one.', sources: [], timestamp: 2 },
          ],
        })}
        engineerMode={false}
      />,
    );
    await userEvent.click(screen.getByTestId('answer-versions-open'));
    expect(screen.getByTestId('answer-versions-position')).toHaveTextContent('2/2');
    expect(screen.getByTestId('answer-version-content')).toHaveTextContent('Newer one.');
    await userEvent.click(screen.getByRole('button', { name: 'Older answer' }));
    expect(screen.getByTestId('answer-version-content')).toHaveTextContent('Older one.');
  });

  it('Reformat asks again in the chosen shape or language, replacing this answer', async () => {
    const onReformat = vi.fn();
    render(
      <ChatMessage
        turn={answer()}
        engineerMode={false}
        regenerateQuestion="How long?"
        onRetry={vi.fn()}
        onReformat={onReformat}
      />,
    );
    await userEvent.click(screen.getByTestId('chat-message-reformat'));
    await userEvent.click(screen.getByTestId('chat-message-reformat-table'));
    expect(onReformat).toHaveBeenCalledWith('How long?', undefined, { answerFormat: 'table' }, 'a1');

    await userEvent.click(screen.getByTestId('chat-message-reformat'));
    await userEvent.type(screen.getByTestId('chat-message-reformat-language'), 'German{Enter}');
    expect(onReformat).toHaveBeenLastCalledWith('How long?', undefined, { answerLanguage: 'German' }, 'a1');
  });

  it('the review panel lists answers worth a second look', async () => {
    const onOpen = vi.fn();
    render(
      <ReviewPanel
        answers={[
          {
            conversationId: 'c1',
            conversationTitle: 'Leave',
            workspace: '',
            question: 'How long?',
            turn: answer({ uncited: true }),
          },
          { conversationId: 'c1', conversationTitle: 'Leave', workspace: '', question: 'Fine?', turn: answer({ id: 'a2' }) },
        ]}
        onOpen={onOpen}
      />,
    );
    expect(screen.getAllByTestId('review-row')).toHaveLength(1);
    expect(screen.getByTestId('review-reasons')).toHaveTextContent('no sources cited');
    await userEvent.click(screen.getByTestId('review-open'));
    expect(onOpen).toHaveBeenCalledWith('c1', 'a1');
  });

  it('conversations are labelled and the list filtered by a label', async () => {
    const onSetLabels = vi.fn();
    render(
      <ConversationList
        conversations={[
          { id: 'c1', title: 'Leave', updatedAt: 2, turnCount: 2, labels: ['hr'] },
          { id: 'c2', title: 'Servers', updatedAt: 1, turnCount: 2 },
        ]}
        activeId={null}
        onSwitch={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        onSetLabels={onSetLabels}
      />,
    );
    await userEvent.click(screen.getByTestId('conversation-label-filter'));
    expect(screen.getAllByTestId('conversation-select')).toHaveLength(1);
    await userEvent.click(screen.getByTestId('conversation-label-filter'));
    expect(screen.getAllByTestId('conversation-select')).toHaveLength(2);

    await userEvent.click(screen.getByRole('button', { name: 'Label Servers' }));
    await userEvent.type(screen.getByTestId('conversation-labels-input'), 'ops, infra{Enter}');
    expect(onSetLabels).toHaveBeenCalledWith('c2', ['ops', ' infra']);
  });

  it('a summary of one document can be limited to a page range', async () => {
    const onRun = vi.fn();
    render(<SummaryDialog filenames={['a.pdf']} onRun={onRun} onClose={vi.fn()} />);
    await userEvent.type(screen.getByTestId('summary-page-min'), '3');
    await userEvent.type(screen.getByTestId('summary-page-max'), '5');
    await userEvent.click(screen.getByTestId('summary-run'));
    expect(onRun).toHaveBeenCalledWith(undefined, { min: 3, max: 5 });
    onRun.mockClear();
    render(<SummaryDialog filenames={['a.pdf', 'b.pdf']} onRun={onRun} onClose={vi.fn()} />);
    expect(screen.getAllByTestId('summary-pages')).toHaveLength(1);
  });

  it('a duplicate document can keep itself and delete the copies', async () => {
    const onDeleteDuplicates = vi.fn().mockResolvedValue(undefined);
    render(
      <DocumentCard
        card={{
          key: 'doc:a.md',
          filename: 'a.md',
          status: 'indexed',
          detail: '1 chunk',
          pct: 100,
          tags: [],
          sameAs: ['copy.md'],
        }}
        readOnly={false}
        confirming={false}
        deleting={false}
        selectable={false}
        selected={false}
        onToggleSelect={vi.fn()}
        onAskDelete={vi.fn()}
        onCancelDelete={vi.fn()}
        onConfirmDelete={vi.fn()}
        onDeleteDuplicates={onDeleteDuplicates}
      />,
    );
    await userEvent.click(screen.getByTestId('corpus-keep-this'));
    await userEvent.click(screen.getByTestId('corpus-keep-this-confirm'));
    expect(onDeleteDuplicates).toHaveBeenCalledWith(['copy.md']);
  });
});
