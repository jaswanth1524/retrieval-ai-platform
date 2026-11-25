import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import Composer from '../../src/components/Composer';
import type { PublicConfigResponse } from '../../src/api/types';

function base() {
  return {
    onSubmit: vi.fn(),
    disabled: false,
    pending: false,
    onCancel: vi.fn(),
    scopeLabel: 'Some documents',
    indexedFilenames: ['a.txt', 'b.txt', 'c.txt'],
    onSelectedFilenamesChange: vi.fn(),
    config: { llm_provider: 'ollama', ollama_available: true } as PublicConfigResponse,
    provider: 'ollama' as const,
    onProviderChange: vi.fn(),
    providerLabel: 'llama3.1:8b',
  };
}

function Scoped({ initial }: { initial: string[] }) {
  const [files, setFiles] = useState(initial);
  return <Composer {...base()} selectedFilenames={files} onSelectedFilenamesChange={setFiles} />;
}

describe('pass 21: composer scope chips', () => {
  it('moves focus to the next chip, then to the question box, as chips are removed', () => {
    render(<Scoped initial={['a.txt', 'b.txt']} />);
    fireEvent.click(screen.getByRole('button', { name: 'Stop limiting to a.txt' }));
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Stop limiting to b.txt' }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop limiting to b.txt' }));
    expect(document.activeElement).toBe(screen.getByTestId('question-textarea'));
  });

  it('counts a long list of tags instead of a chip each, and keeps the document chips', () => {
    const tags = Array.from({ length: 12 }, (_, i) => `tag${i}`);
    render(
      <Composer
        {...base()}
        selectedFilenames={['a.txt']}
        selectedTags={tags}
        onSelectedTagsChange={vi.fn()}
      />,
    );
    const chips = screen.getByTestId('composer-scope-chips');
    expect(chips).toHaveTextContent('12 tags');
    expect(screen.getByRole('button', { name: 'Stop limiting to a.txt' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Stop limiting to the tag tag0' })).toBeNull();
  });
});

describe('pass 21: a queued send', () => {
  it('is dropped when the workspace changes, its question kept in the box', async () => {
    const props = { ...base(), selectedFilenames: [] as string[], sendHeld: true, sendQueueable: true };
    const { rerender } = render(<Composer {...props} workspace="" />);
    await userEvent.type(screen.getByTestId('question-textarea'), 'What is the refund window?');
    await userEvent.keyboard('{Enter}');
    expect(screen.getByTestId('question-queued')).toBeInTheDocument();

    rerender(<Composer {...props} workspace="legal" />);
    rerender(<Composer {...props} workspace="legal" sendHeld={false} sendQueueable={false} />);
    expect(props.onSubmit).not.toHaveBeenCalled();
    expect(screen.getByTestId('question-textarea')).toHaveValue('What is the refund window?');
  });
});
