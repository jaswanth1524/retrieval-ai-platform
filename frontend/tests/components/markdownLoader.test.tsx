import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';

// Each test imports fresh modules, so the loader starts with nothing loaded.
beforeEach(() => {
  vi.resetModules();
  vi.doUnmock('../../src/components/Markdown');
});

const answer: ChatTurn = {
  id: 'a',
  role: 'assistant',
  content: '**Use docker** [1].',
  sources: [
    { source_number: 1, filename: 'guide.md', page: 1, section: 'Setup', chunk_id: 'c1', text: 't' },
  ],
  timestamp: 0,
  timings: null,
  traceId: null,
};

describe('markdown renderer loading', () => {
  it('shows the answer as plain text until the renderer arrives, then as markdown', async () => {
    const { default: ChatMessage } = await import('../../src/components/ChatMessage');
    render(<ChatMessage turn={answer} engineerMode={false} />);

    // The raw answer — never the "[1](#docrag-cite-1)" link syntax it is rewritten to.
    expect(screen.getByText('**Use docker** [1].')).toBeInTheDocument();

    expect((await screen.findByText('Use docker')).tagName).toBe('STRONG');
    expect(screen.getByTestId('inline-citation')).toHaveTextContent('1');
  });

  it('falls back to plain text when the renderer chunk cannot load', async () => {
    vi.doMock('../../src/components/Markdown', () => {
      throw new Error('Failed to fetch dynamically imported module');
    });
    const { loadMarkdown } = await import('../../src/components/markdownLoader');
    const { default: PlainAnswer } = await import('../../src/components/PlainAnswer');

    await expect(loadMarkdown()).resolves.toBe(PlainAnswer);

    // A later answer tries again: one failure doesn't mean plain text until a reload.
    vi.doUnmock('../../src/components/Markdown');
    await expect(loadMarkdown()).resolves.not.toBe(PlainAnswer);
  });
});
