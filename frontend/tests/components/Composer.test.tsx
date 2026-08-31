import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import Composer from '../../src/components/Composer';
import type { PublicConfigResponse } from '../../src/api/types';

function makeConfig(overrides: Partial<PublicConfigResponse> = {}): PublicConfigResponse {
  return {
    qdrant_collection: 'docrag_documents',
    dense_embedding_model: 'BAAI/bge-small-en-v1.5',
    sparse_embedding_model: 'Qdrant/BM25',
    reranker_model: 'jinaai/jina-reranker-v2-base-multilingual',
    embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
    llm_provider: 'ollama',
    llm_model: 'llama3.1:8b',
    llm_temperature: 0,
    openai_model: 'gpt-4o-mini',
    openai_available: false,
    ollama_available: true,
    rrf_k: 60,
    dense_retrieval_limit: 50,
    sparse_retrieval_limit: 50,
    fused_top_n: 50,
    rerank_top_k: 8,
    max_context_chunks: 6,
    rerank_min_score: 0.3,
    max_upload_bytes: 52428800,
    rerank_top_k_limit: 50,
    max_context_chunks_limit: 20,
    llm_temperature_max: 2,
    ...overrides,
  };
}

function baseProps(overrides: Partial<Parameters<typeof Composer>[0]> = {}) {
  return {
    onSubmit: vi.fn(),
    disabled: false,
    pending: false,
    onCancel: vi.fn(),
    scopeLabel: 'All 2 documents',
    indexedFilenames: ['a.txt', 'b.txt'],
    selectedFilenames: [] as string[],
    onSelectedFilenamesChange: vi.fn(),
    config: makeConfig(),
    provider: 'ollama' as const,
    onProviderChange: vi.fn(),
    providerLabel: 'llama3.1:8b',
    ...overrides,
  };
}

describe('Composer', () => {
  it('submits and clears the value on Enter', async () => {
    const props = baseProps();
    render(<Composer {...props} />);

    const textarea = screen.getByTestId('question-textarea');
    await userEvent.type(textarea, 'What is the refund window?');
    await userEvent.keyboard('{Enter}');

    expect(props.onSubmit).toHaveBeenCalledWith('What is the refund window?');
    expect(textarea).toHaveValue('');
  });

  it('inserts a newline instead of submitting on Shift+Enter', async () => {
    const props = baseProps();
    render(<Composer {...props} />);

    const textarea = screen.getByTestId('question-textarea');
    await userEvent.type(textarea, 'line one{Shift>}{Enter}{/Shift}line two');

    expect(props.onSubmit).not.toHaveBeenCalled();
    expect(textarea).toHaveValue('line one\nline two');
  });

  it('does not submit whitespace-only input', async () => {
    const props = baseProps();
    render(<Composer {...props} />);

    await userEvent.type(screen.getByTestId('question-textarea'), '   {Enter}');

    expect(props.onSubmit).not.toHaveBeenCalled();
  });

  it('disables the textarea and submit button when disabled', () => {
    render(<Composer {...baseProps({ disabled: true })} />);

    expect(screen.getByTestId('question-textarea')).toBeDisabled();
    expect(screen.getByTestId('question-submit')).toBeDisabled();
  });

  it('caps input length to match the backend max_length', () => {
    render(<Composer {...baseProps()} />);

    expect(screen.getByTestId('question-textarea')).toHaveAttribute('maxLength', '4000');
  });

  it('renders a hint line when provided', () => {
    render(<Composer {...baseProps({ disabled: true, hint: 'Upload a document to start.' })} />);

    expect(screen.getByTestId('question-hint')).toHaveTextContent('Upload a document to start.');
  });

  it('renders no hint line when hint is omitted', () => {
    render(<Composer {...baseProps()} />);

    expect(screen.queryByTestId('question-hint')).not.toBeInTheDocument();
  });

  it('shows Stop instead of Ask while pending, and fires onCancel', async () => {
    const props = baseProps({ pending: true });
    render(<Composer {...props} />);

    const button = screen.getByTestId('question-submit');
    expect(button).toHaveTextContent('Stop');
    expect(button).toBeEnabled();

    await userEvent.click(button);
    expect(props.onCancel).toHaveBeenCalledOnce();
  });

  describe('scope popover', () => {
    it('opens on click and lists every indexed filename', async () => {
      render(<Composer {...baseProps()} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));

      expect(screen.getAllByTestId('scope-item')).toHaveLength(2);
    });

    it('toggling a filename calls onSelectedFilenamesChange', async () => {
      const props = baseProps();
      render(<Composer {...props} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));
      const items = screen.getAllByTestId('scope-item');
      await userEvent.click(items[0].querySelector('input') as HTMLInputElement);

      expect(props.onSelectedFilenamesChange).toHaveBeenCalledWith(['a.txt']);
    });

    it('closes on outside click', async () => {
      render(
        <div>
          <Composer {...baseProps()} />
          <button type="button">outside</button>
        </div>,
      );

      await userEvent.click(screen.getByTestId('composer-scope-button'));
      expect(screen.getByTestId('composer-scope-popover')).toBeInTheDocument();

      await userEvent.click(screen.getByText('outside'));
      expect(screen.queryByTestId('composer-scope-popover')).not.toBeInTheDocument();
    });

    it('closes on Escape', async () => {
      render(<Composer {...baseProps()} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));
      await userEvent.keyboard('{Escape}');

      expect(screen.queryByTestId('composer-scope-popover')).not.toBeInTheDocument();
    });

    it('shows no filter box for a small corpus', async () => {
      render(<Composer {...baseProps()} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));

      expect(screen.queryByTestId('composer-scope-search')).not.toBeInTheDocument();
    });

    it('filters the checkbox list by a substring match once the corpus is large enough', async () => {
      const filenames = ['contract.pdf', 'invoice.txt', 'README.md', 'notes.md', 'plan.md', 'todo.md'];
      render(<Composer {...baseProps({ indexedFilenames: filenames })} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));
      expect(screen.getAllByTestId('scope-item')).toHaveLength(6);

      await userEvent.type(screen.getByTestId('composer-scope-search'), 'read');

      const items = screen.getAllByTestId('scope-item');
      expect(items).toHaveLength(1);
      expect(items[0]).toHaveTextContent('README.md');
    });

    it('keeps an already-selected filename visible even when it does not match the filter', async () => {
      // Hard requirement: the checkbox list is the only way to unselect a filename —
      // filtering it out of view would make it permanently unselectable.
      const filenames = ['contract.pdf', 'invoice.txt', 'README.md', 'notes.md', 'plan.md', 'todo.md'];
      const props = baseProps({ indexedFilenames: filenames, selectedFilenames: ['plan.md'] });
      render(<Composer {...props} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));
      await userEvent.type(screen.getByTestId('composer-scope-search'), 'read');

      const items = screen.getAllByTestId('scope-item');
      const names = items.map((item) => item.textContent);
      expect(names.some((text) => text?.includes('README.md'))).toBe(true);
      expect(names.some((text) => text?.includes('plan.md'))).toBe(true);
      expect(items).toHaveLength(2);

      // And it must still be togglable from that filtered view.
      const planItem = items.find((item) => item.textContent?.includes('plan.md'));
      await userEvent.click(planItem!.querySelector('input') as HTMLInputElement);
      expect(props.onSelectedFilenamesChange).toHaveBeenCalledWith([]);
    });

    it('resets the filter each time the popover reopens', async () => {
      const filenames = ['contract.pdf', 'invoice.txt', 'README.md', 'notes.md', 'plan.md', 'todo.md'];
      render(<Composer {...baseProps({ indexedFilenames: filenames })} />);

      await userEvent.click(screen.getByTestId('composer-scope-button'));
      await userEvent.type(screen.getByTestId('composer-scope-search'), 'read');
      await userEvent.click(screen.getByTestId('composer-scope-button')); // close
      await userEvent.click(screen.getByTestId('composer-scope-button')); // reopen

      expect(screen.getByTestId('composer-scope-search')).toHaveValue('');
      expect(screen.getAllByTestId('scope-item')).toHaveLength(6);
    });

    it('is disabled when there are no indexed documents', () => {
      render(<Composer {...baseProps({ indexedFilenames: [] })} />);

      expect(screen.getByTestId('composer-scope-button')).toBeDisabled();
    });
  });

  describe('provider popover', () => {
    it('opens and shows both providers with reachability hints', async () => {
      render(<Composer {...baseProps({ config: makeConfig({ ollama_available: false, openai_available: true }) })} />);

      await userEvent.click(screen.getByTestId('composer-provider-button'));

      expect(screen.getByTestId('provider-ollama')).toBeDisabled();
      expect(screen.getByTestId('provider-openai')).toBeEnabled();
      expect(screen.getByText(/isn.t reachable/)).toBeInTheDocument();
    });

    it('selecting a provider calls onProviderChange', async () => {
      const props = baseProps({ config: makeConfig({ openai_available: true }) });
      render(<Composer {...props} />);

      await userEvent.click(screen.getByTestId('composer-provider-button'));
      await userEvent.click(screen.getByTestId('provider-openai'));

      expect(props.onProviderChange).toHaveBeenCalledWith('openai');
    });
  });
  describe('popover accessibility', () => {
    it('announces popover state on the trigger and labels the popover itself', async () => {
      render(<Composer {...baseProps({ config: makeConfig({ openai_available: true }) })} />);

      const scopeButton = screen.getByTestId('composer-scope-button');
      expect(scopeButton).toHaveAttribute('aria-haspopup', 'true');
      expect(scopeButton).toHaveAttribute('aria-expanded', 'false');

      await userEvent.click(scopeButton);

      expect(scopeButton).toHaveAttribute('aria-expanded', 'true');
      const popover = screen.getByTestId('composer-scope-popover');
      expect(popover).toHaveAttribute('role', 'group');
      expect(popover).toHaveAccessibleName('Search scope');
      // The trigger points at the popover it controls.
      expect(scopeButton).toHaveAttribute('aria-controls', popover.id);
    });

    it('marks the provider popover as a radiogroup', async () => {
      render(<Composer {...baseProps({ config: makeConfig({ openai_available: true }) })} />);

      await userEvent.click(screen.getByTestId('composer-provider-button'));

      const popover = screen.getByTestId('composer-provider-popover');
      expect(popover).toHaveAttribute('role', 'radiogroup');
      expect(popover).toHaveAccessibleName('Generation provider');
    });

    it('Escape closes the popover and returns focus to its trigger', async () => {
      // Without this, escaping drops focus to document.body and a keyboard user is
      // stranded at the top of the page.
      render(<Composer {...baseProps({ config: makeConfig({ openai_available: true }) })} />);

      const scopeButton = screen.getByTestId('composer-scope-button');
      await userEvent.click(scopeButton);
      expect(screen.getByTestId('composer-scope-popover')).toBeInTheDocument();

      await userEvent.keyboard('{Escape}');

      expect(screen.queryByTestId('composer-scope-popover')).not.toBeInTheDocument();
      expect(scopeButton).toHaveFocus();
    });
  });
});
