import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ChatToolbar from '../../src/components/ChatToolbar';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { chatToMarkdown } from '../../src/utils/exportChat';

function makeTurn(overrides: Partial<ChatTurn>): ChatTurn {
  return {
    id: 'turn-1',
    role: 'user',
    content: 'Hello',
    sources: [],
    timestamp: 0,
    timings: null,
    traceId: null,
    ...overrides,
  };
}

describe('chatToMarkdown', () => {
  it('renders speaker headings, content, and sources for each turn', () => {
    const turns: ChatTurn[] = [
      makeTurn({ role: 'user', content: 'What is DocRAG?', timestamp: 0 }),
      makeTurn({
        role: 'assistant',
        content: 'A document Q&A system [1].',
        timestamp: 0,
        sources: [
          {
            source_number: 1,
            filename: 'guide.md',
            page: 1,
            section: 'Intro',
            chunk_id: 'c1',
            text: 'excerpt',
          },
        ],
      }),
    ];

    const markdown = chatToMarkdown(turns);

    expect(markdown).toContain('## You');
    expect(markdown).toContain('What is DocRAG?');
    expect(markdown).toContain('## DocRAG');
    expect(markdown).toContain('A document Q&A system [1].');
    expect(markdown).toContain('[1] guide.md · p.1 · Intro');
  });
});

describe('ChatToolbar', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders nothing when there are no turns', () => {
    const { container } = render(<ChatToolbar turns={[]} onClear={vi.fn()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it('clicking Clear chat shows an inline confirm, and confirming calls onClear', async () => {
    const onClear = vi.fn();
    render(<ChatToolbar turns={[makeTurn({})]} onClear={onClear} />);

    await userEvent.click(screen.getByTestId('chat-toolbar-clear'));
    expect(screen.getByText('Clear chat?')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('chat-toolbar-clear-confirm'));

    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it('cancelling the clear confirm does not call onClear', async () => {
    const onClear = vi.fn();
    render(<ChatToolbar turns={[makeTurn({})]} onClear={onClear} />);

    await userEvent.click(screen.getByTestId('chat-toolbar-clear'));
    await userEvent.click(screen.getByText('Cancel'));

    expect(onClear).not.toHaveBeenCalled();
    expect(screen.queryByText('Clear chat?')).not.toBeInTheDocument();
  });

  it('exporting downloads a file via a Blob URL', async () => {
    const createObjectURL = vi.fn().mockReturnValue('blob:mock');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL });
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    render(<ChatToolbar turns={[makeTurn({})]} onClear={vi.fn()} />);
    await userEvent.click(screen.getByTestId('chat-toolbar-export-markdown'));

    expect(createObjectURL).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock');

    clickSpy.mockRestore();
  });
});
