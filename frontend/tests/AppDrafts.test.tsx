import { act, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

// The full history lives in IndexedDB, which jsdom lacks: stand in for it, with a load
// the test releases when it chooses.
let finishLoad: ((value: unknown) => void) | undefined;
vi.mock('../src/hooks/chatStore', () => ({
  chatStoreAvailable: () => true,
  loadChatStore: () =>
    new Promise((resolve) => {
      finishLoad = resolve;
    }),
  saveChatStore: async () => true,
  clearChatStore: async () => true,
  resetChatStoreConnection: () => {},
}));

const App = (await import('../src/App')).default;

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

it("keeps a draft of a conversation only IndexedDB holds until the history has loaded", async () => {
  // localStorage's compact copy can lack a conversation the full history still has.
  localStorage.setItem(
    'docrag-drafts',
    JSON.stringify({ 'idb-only': { text: 'half-written question', at: Date.now() } }),
  );
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/config')) {
        return new Response(
          JSON.stringify({ llm_provider: 'ollama', ollama_available: true, openai_available: false }),
          { status: 200 },
        );
      }
      if (url.endsWith('/documents')) {
        return new Response(JSON.stringify({ filenames: ['g.md'], chunk_counts: { 'g.md': 1 } }), {
          status: 200,
        });
      }
      throw new Error(`unexpected fetch: ${url}`);
    }),
  );

  render(<App />);
  await screen.findByText('api ok');
  // Before the load: the conversation isn't known yet, and its draft must survive that.
  expect(JSON.parse(localStorage.getItem('docrag-drafts') ?? '{}')).toHaveProperty('idb-only');

  await act(async () => {
    finishLoad?.({
      version: 2,
      activeConversationId: 'idb-only',
      conversations: [
        {
          id: 'idb-only',
          title: 'Old one',
          createdAt: 1,
          updatedAt: 2,
          turns: [{ id: 't1', role: 'user', content: 'hello', sources: [], timestamp: 1 }],
        },
      ],
    });
  });

  await waitFor(() => expect(screen.getAllByText('Old one').length).toBeGreaterThan(0));
  expect(JSON.parse(localStorage.getItem('docrag-drafts') ?? '{}')).toHaveProperty('idb-only');
});
