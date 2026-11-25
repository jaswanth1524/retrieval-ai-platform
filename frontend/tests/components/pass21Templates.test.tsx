import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Composer from '../../src/components/Composer';
import type { PublicConfigResponse } from '../../src/api/types';
import { firstPlaceholder, readTemplates, saveTemplate } from '../../src/utils/questionTemplates';

function props() {
  return {
    onSubmit: vi.fn(),
    disabled: false,
    pending: false,
    onCancel: vi.fn(),
    scopeLabel: 'All documents',
    indexedFilenames: ['a.txt'],
    selectedFilenames: [] as string[],
    onSelectedFilenamesChange: vi.fn(),
    config: { llm_provider: 'ollama', ollama_available: true } as PublicConfigResponse,
    provider: 'ollama' as const,
    onProviderChange: vi.fn(),
    providerLabel: 'llama3.1:8b',
  };
}

afterEach(() => localStorage.clear());

describe('pass 21: question templates', () => {
  it('are kept in this browser, newest first, one per name', () => {
    saveTemplate('Topic', 'What does {document} say about {topic}?');
    saveTemplate('Dates', 'When does {thing} start?');
    saveTemplate('topic', 'Summarize {topic}.');
    expect(readTemplates().map((t) => t.name)).toEqual(['topic', 'Dates']);
    expect(firstPlaceholder('When does {thing} start?')).toEqual([10, 17]);
    expect(firstPlaceholder('No blanks.')).toBeNull();
  });

  it('save the question in the box and put one back with its first blank selected', async () => {
    render(<Composer {...props()} />);
    const box = screen.getByTestId('question-textarea') as HTMLTextAreaElement;
    await userEvent.type(box, 'When does {{thing} start?');
    await userEvent.click(screen.getByTestId('composer-templates-button'));
    await userEvent.type(screen.getByTestId('template-name'), 'Start date');
    await userEvent.click(screen.getByTestId('template-save'));
    expect(readTemplates()).toEqual([{ name: 'Start date', text: 'When does {thing} start?' }]);

    await userEvent.clear(box);
    await userEvent.click(screen.getByTestId('composer-templates-button'));
    await userEvent.click(screen.getByTestId('template-insert'));
    expect(box.value).toBe('When does {thing} start?');
    expect(document.activeElement).toBe(box);
    expect([box.selectionStart, box.selectionEnd]).toEqual([10, 17]);
  });
});
