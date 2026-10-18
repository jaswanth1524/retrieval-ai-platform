import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';

// Stands in for the corpus panel to see the summarize handler App gives it on each render.
const summarizeHandlers: unknown[] = [];
vi.mock('../src/components/CorpusPanel', () => ({
  default: (props: { onSummarize?: unknown }) => {
    summarizeHandlers.push(props.onSummarize);
    return null;
  },
}));

const App = (await import('../src/App')).default;

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

it('gives the document cards one summarize handler, whatever the provider or scope', async () => {
  // The model label (part of what a question is asked with) arrives with /config; the
  // handler used to change with it, and with every scope or provider change, re-rendering
  // every memoized card of a large corpus.
  let answerConfig!: () => void;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/config')) {
        return new Promise<Response>((resolve) => {
          answerConfig = () =>
            resolve(
              new Response(
                JSON.stringify({ llm_provider: 'ollama', llm_model: 'llama3.1:8b', ollama_available: true }),
                { status: 200 },
              ),
            );
        });
      }
      if (url.endsWith('/documents')) {
        return new Response(JSON.stringify({ filenames: ['g.md'], chunk_counts: { 'g.md': 1 } }), { status: 200 });
      }
      if (url.endsWith('/access')) return new Response(JSON.stringify({ access: 'open' }), { status: 200 });
      throw new Error(`unexpected fetch: ${url}`);
    }),
  );

  render(<App />);
  await userEvent.click(screen.getByTestId('rail-corpus'));
  const before = summarizeHandlers.at(-1);
  expect(before).toBeTypeOf('function');

  await act(async () => answerConfig());
  await screen.findByText('api ok');
  await waitFor(() => expect(summarizeHandlers.length).toBeGreaterThan(1));
  expect(new Set(summarizeHandlers.filter(Boolean)).size).toBe(1);
});
