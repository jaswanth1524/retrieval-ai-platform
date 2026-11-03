import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ChatMessage, { type ChatTurn } from '../../src/components/ChatMessage';
import { useDialog } from '../../src/hooks/useDialog';
import { copyText } from '../../src/utils/clipboard';

function Dialog({ open }: { open: boolean }) {
  const ref = useDialog(open, () => {});
  if (!open) return null;
  return (
    <div ref={ref}>
      <button type="button">first</button>
      <button type="button">last</button>
    </div>
  );
}

const errorTurn = (id: string): ChatTurn => ({
  id,
  role: 'error',
  content: 'The model did not answer.',
  sources: [],
  timestamp: Date.now(),
  timings: null,
  traceId: null,
});

describe('pass 14', () => {
  it('Tab from outside an open dialog goes back into it, not past it', () => {
    render(<Dialog open />);
    (document.activeElement as HTMLElement | null)?.blur();
    expect(document.activeElement).toBe(document.body);

    fireEvent.keyDown(document, { key: 'Tab' });

    expect(screen.getByText('first')).toHaveFocus();
  });

  it('an error is announced when it happens, not again when its conversation is reopened', () => {
    const turn = errorTurn('err-1');
    const first = render(<ChatMessage turn={turn} engineerMode={false} />);
    expect(screen.getByRole('alert')).toHaveTextContent('The model did not answer.');
    first.unmount();

    // The thread remounts when the conversation is opened again.
    render(<ChatMessage turn={turn} engineerMode={false} />);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByText('The model did not answer.')).toBeInTheDocument();
  });

  it('copies on plain HTTP too, where there is no Clipboard API', async () => {
    const clipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    const exec = vi.fn(() => true);
    Object.defineProperty(document, 'execCommand', { value: exec, configurable: true });
    try {
      await expect(copyText('Ten days [1].')).resolves.toBe(true);
      expect(exec).toHaveBeenCalledWith('copy');
    } finally {
      if (clipboard) Object.defineProperty(navigator, 'clipboard', clipboard);
    }
  });

  it('says how many picked passages an answer went without', () => {
    render(
      <ChatMessage
        turn={{ ...errorTurn('a-1'), role: 'assistant', content: 'Alpha [1].', passagesLeftOut: 2 }}
        engineerMode={false}
      />,
    );
    expect(screen.getByText(/2 of the passages you picked weren't read/)).toBeInTheDocument();
  });
});

describe('pass 14 saved answers across workspaces', () => {
  it("never flags another workspace's sources as deleted", async () => {
    const { default: SavedPanel } = await import('../../src/components/SavedPanel');
    const source = {
      source_number: 1,
      filename: 'policy.md',
      page: 1,
      section: 'Intro',
      chunk_id: 'c1',
      text: 'Alpha.',
    };
    const turn = (id: string): ChatTurn => ({
      ...errorTurn(id),
      role: 'assistant',
      content: 'Alpha [1].',
      sources: [source as ChatTurn['sources'][number]],
      saved: true,
    });
    render(
      <SavedPanel
        answers={[
          { conversationId: 'a', conversationTitle: 'Here', workspace: '', turn: turn('t1'), question: 'Q?' },
          { conversationId: 'b', conversationTitle: 'Legal', workspace: 'legal', turn: turn('t2'), question: 'Q?' },
        ]}
        onOpen={() => {}}
        onUnsave={() => {}}
        sourceChangeOf={() => 'deleted'}
        workspace=""
      />,
    );
    // Checked only for the answer from the listed workspace.
    expect(screen.getAllByText(/deleted: policy.md/)).toHaveLength(1);
  });
});
