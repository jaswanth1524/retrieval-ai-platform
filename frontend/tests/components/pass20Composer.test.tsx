import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import Composer from '../../src/components/Composer';
import type { PublicConfigResponse } from '../../src/api/types';

function props(overrides: Partial<Parameters<typeof Composer>[0]> = {}) {
  return {
    onSubmit: vi.fn(),
    disabled: false,
    pending: false,
    onCancel: vi.fn(),
    scopeLabel: 'All 2 documents',
    indexedFilenames: ['a.txt', 'reports/b.txt'],
    selectedFilenames: [] as string[],
    onSelectedFilenamesChange: vi.fn(),
    config: { llm_provider: 'ollama', ollama_available: true } as PublicConfigResponse,
    provider: 'ollama' as const,
    onProviderChange: vi.fn(),
    providerLabel: 'llama3.1:8b',
    ...overrides,
  };
}

describe('pass 20: composer', () => {
  it('puts the last question back on Up in an empty box, and only then', () => {
    render(<Composer {...props({ recallQuestion: 'How long is leave?' })} />);
    const box = screen.getByTestId('question-textarea') as HTMLTextAreaElement;
    fireEvent.keyDown(box, { key: 'ArrowUp' });
    expect(box.value).toBe('How long is leave?');
    fireEvent.change(box, { target: { value: 'typed' } });
    fireEvent.keyDown(box, { key: 'ArrowUp' });
    expect(box.value).toBe('typed');
  });

  it('lists the scope as chips that each take one document or tag away', () => {
    const onFiles = vi.fn();
    const onTags = vi.fn();
    render(
      <Composer
        {...props({
          selectedFilenames: ['a.txt', 'reports/b.txt'],
          onSelectedFilenamesChange: onFiles,
          selectedTags: ['hr'],
          onSelectedTagsChange: onTags,
        })}
      />,
    );
    expect(screen.getByTestId('composer-scope-chips')).toHaveTextContent('b.txt');
    fireEvent.click(screen.getByRole('button', { name: 'Stop limiting to reports/b.txt' }));
    expect(onFiles).toHaveBeenCalledWith(['a.txt']);
    fireEvent.click(screen.getByRole('button', { name: 'Stop limiting to the tag hr' }));
    expect(onTags).toHaveBeenCalledWith([]);
  });

  it('shows no chips without a scope', () => {
    render(<Composer {...props()} />);
    expect(screen.queryByTestId('composer-scope-chips')).toBeNull();
  });
});

describe('pass 20: continuing a stopped answer', () => {
  it('offers Continue on a stopped or broken-off answer with a trace', async () => {
    const { default: ChatMessage } = await import('../../src/components/ChatMessage');
    const onContinue = vi.fn();
    for (const flag of [{ stopped: true }, { incomplete: true }]) {
      const { unmount } = render(
        <ChatMessage
          turn={{
            id: 'a1',
            role: 'assistant',
            content: 'Alpha is',
            sources: [],
            timestamp: 1,
            timings: null,
            traceId: 't1',
            ...flag,
          }}
          engineerMode={false}
          regenerateQuestion="What is alpha?"
          onRetry={vi.fn()}
          onContinue={onContinue}
        />,
      );
      fireEvent.click(screen.getByTestId('chat-message-continue'));
      unmount();
    }
    expect(onContinue).toHaveBeenCalledTimes(2);
  });
});
