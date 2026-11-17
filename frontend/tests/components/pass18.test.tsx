import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Composer from '../../src/components/Composer';
import type { PublicConfigResponse } from '../../src/api/types';
import RecentlyDeleted from '../../src/components/RecentlyDeleted';
import type { ChatTurn } from '../../src/components/ChatMessage';
import type { SavedAnswer } from '../../src/hooks/useChat';
import { withConversationRestored } from '../../src/hooks/chatPersistence';
import { asVersion } from '../../src/utils/answerVersions';
import { usageTotal } from '../../src/utils/deepLink';
import { checkable, savedCheckRequest } from '../../src/utils/savedCheck';
import { setWorkspace } from '../../src/api/client';
import { clearRecentSearches, readRecentSearches, rememberSearch } from '../../src/utils/searchHistory';
import { addToTrash, readTrash } from '../../src/utils/trash';

afterEach(() => localStorage.clear());

function makeConfig(): PublicConfigResponse {
  return {
    qdrant_collection: 'docrag_documents',
    dense_embedding_model: 'BAAI/bge-small-en-v1.5',
    sparse_embedding_model: 'Qdrant/BM25',
    reranker_model: 'jinaai/jina-reranker-v2-base-multilingual',
    embedding_model_tag: 'tag',
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
    rerank_min_score: 0.15,
    max_upload_bytes: 52428800,
    rerank_top_k_limit: 50,
    max_context_chunks_limit: 20,
    llm_temperature_max: 2.0,
    feedback_enabled: false,
  };
}

function answer(id: string, extra: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id,
    role: 'assistant',
    content: 'Alpha [1].',
    sources: [],
    timings: null,
    traceId: null,
    timestamp: 1,
    ...extra,
  } as ChatTurn;
}

describe('composer prefill', () => {
  it('is added under a draft, not over it', async () => {
    const props = {
      onSubmit: vi.fn(),
      disabled: false,
      pending: false,
      onCancel: vi.fn(),
      scopeLabel: 'all documents',
      indexedFilenames: ['a.txt'],
      selectedFilenames: [],
      onSelectedFilenamesChange: vi.fn(),
      config: makeConfig(),
      provider: 'ollama' as const,
      onProviderChange: vi.fn(),
      providerLabel: 'llama3.1:8b',
    };
    const { rerender } = render(<Composer {...props} prefill={null} />);
    const box = screen.getByTestId('question-textarea') as HTMLTextAreaElement;
    await userEvent.type(box, 'My half-written question');
    rerender(<Composer {...props} prefill={{ text: 'About “a quote”: ', nonce: 1 }} />);
    expect(box.value).toBe('My half-written question\nAbout “a quote”: ');
    // An empty box simply takes it.
    await userEvent.clear(box);
    rerender(<Composer {...props} prefill={{ text: 'Example question?', nonce: 2 }} />);
    expect(box.value).toBe('Example question?');
  });
});

describe('usage totals', () => {
  it('count the answers Regenerate replaced', () => {
    const old = answer('a1', { usage: { prompt: 100, completion: 20, costUsd: 0.01 } });
    const current = answer('a2', {
      usage: { prompt: 50, completion: 10, costUsd: 0.005 },
      versions: [asVersion(old)],
    });
    expect(asVersion(old).usage).toEqual(old.usage);
    expect(usageTotal([current])).toEqual({ tokens: 180, costUsd: 0.015 });
  });
});

describe('recently deleted', () => {
  const conversation = { id: 'c1', title: 'Leave questions', turns: [answer('t1')], createdAt: 1, updatedAt: 2 };

  it('does not offer a conversation that is in the history again', () => {
    addToTrash(conversation as never, Date.now());
    const { rerender } = render(
      <RecentlyDeleted version={0} liveIds={new Set(['c1'])} onRestore={vi.fn()} />,
    );
    expect(screen.queryByTestId('recently-deleted')).toBeNull();
    rerender(<RecentlyDeleted version={0} liveIds={new Set()} onRestore={vi.fn()} />);
    expect(screen.getByText(/Recently deleted \(1\)/)).toBeTruthy();
  });

  it('asks before deleting for good', async () => {
    addToTrash(conversation as never, Date.now());
    render(<RecentlyDeleted version={0} liveIds={new Set()} onRestore={vi.fn()} />);
    await userEvent.click(screen.getByText(/Recently deleted \(1\)/));
    await userEvent.click(screen.getByTestId('recently-deleted-purge'));
    expect(readTrash()).toHaveLength(1);
    await userEvent.click(screen.getByTestId('recently-deleted-purge-confirm'));
    expect(readTrash()).toEqual([]);
  });

  it('a restored conversation counts as used now, so a full history keeps it', () => {
    const storage = { version: 2 as const, conversations: [], activeConversationId: null, deleted: [] };
    const undo = withConversationRestored(storage as never, conversation as never);
    expect(undo.conversations[0].updatedAt).toBe(2);
    const restored = withConversationRestored(storage as never, conversation as never, true);
    expect(restored.conversations[0].updatedAt).toBeGreaterThan(2);
  });
});

describe('saved answer check', () => {
  const cited = { source_number: 1, filename: 'guide.md', page: 1, section: 'S', chunk_id: 'c1', text: '' };
  const saved = (extra: Partial<SavedAnswer> = {}, turn: Partial<ChatTurn> = {}): SavedAnswer => ({
    conversationId: 'c',
    conversationTitle: 'T',
    workspace: '',
    question: 'How long is leave?',
    turn: answer('a', { sources: [cited as never], ...turn }),
    ...extra,
  });

  it('skips summaries and follow-ups, whose question is not what was searched', () => {
    expect(checkable(saved(), '')).toBe(true);
    expect(checkable(saved({ task: true }), '')).toBe(false);
    expect(checkable(saved({}, { followUp: true }), '')).toBe(false);
    expect(checkable(saved(), 'other')).toBe(false);
  });

  it('searches where the question looked', () => {
    expect(savedCheckRequest(saved())).toEqual({ query: 'How long is leave?', rerank_top_k: 8 });
    const narrowed = savedCheckRequest(
      saved({ narrowing: { page_min: 3, chunk_ids: ['x'] } }, { scoped: true }),
    );
    expect(narrowed).toEqual({
      query: 'How long is leave?',
      rerank_top_k: 8,
      page_min: 3,
      filenames: ['guide.md'],
    });
  });
});

describe('recent searches', () => {
  it('can be cleared and never keep a pasted passage', () => {
    rememberSearch('leave policy');
    expect(rememberSearch('x'.repeat(400))).toEqual(['leave policy']);
    expect(clearRecentSearches()).toEqual([]);
    expect(readRecentSearches()).toEqual([]);
  });

  it('are kept per workspace', () => {
    try {
      rememberSearch('leave policy');
      setWorkspace('legal');
      expect(readRecentSearches()).toEqual([]);
      rememberSearch('contract terms');
      expect(readRecentSearches()).toEqual(['contract terms']);
      setWorkspace('');
      expect(readRecentSearches()).toEqual(['leave policy']);
    } finally {
      setWorkspace('');
    }
  });
});
