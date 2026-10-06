import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SearchPanel from '../../src/components/SearchPanel';

const RESULT = {
  rank: 1,
  filename: 'policy.pdf',
  page: 3,
  section: 'Leave',
  chunk_id: 'c1',
  chunk_ordinal: 4,
  text: 'Employees accrue twenty days of leave.',
  retrieval_score: 0.03,
  rerank_score: 0.91,
};

// The real client over a stubbed fetch, so the request the panel sends is checked too.
function stubSearch(status: number, body: unknown): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async () =>
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function setup(props: Partial<Parameters<typeof SearchPanel>[0]> = {}) {
  const handlers = { onOpenSource: vi.fn(), onAuthFailure: vi.fn(() => false) };
  const view = render(
    <SearchPanel filenames={[]} tags={[]} resetKey={0} {...handlers} {...props} />,
  );
  return { ...handlers, ...view };
}

async function search(text: string) {
  await userEvent.type(screen.getByTestId('search-input'), text);
  await userEvent.click(screen.getByTestId('search-submit'));
}

describe('SearchPanel', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('lists ranked passages with their score and opens one in the viewer', async () => {
    stubSearch(200, { results: [RESULT], timings: {} });
    const { onOpenSource } = setup();

    await search('leave');

    expect(await screen.findByText('policy.pdf')).toBeInTheDocument();
    expect(screen.getByText('0.91')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('search-result'));
    expect(onOpenSource).toHaveBeenCalledWith('policy.pdf', 'c1');
  });

  it('searches only the conversation scope', async () => {
    const fetchMock = stubSearch(200, { results: [], timings: {} });
    setup({ filenames: ['policy.pdf'], tags: ['hr'] });

    await search('leave');

    expect(await screen.findByText('No passages matched.')).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/search$/);
    expect(JSON.parse(String(init.body))).toEqual({
      query: 'leave',
      filenames: ['policy.pdf'],
      tags: ['hr'],
    });
  });

  it('says the server is busy on a 429', async () => {
    stubSearch(429, { detail: 'Too many questions.' });
    setup();

    await search('leave');

    expect((await screen.findByTestId('search-error')).textContent).toContain(
      'busy answering questions',
    );
  });

  it('clears when the panel action bumps resetKey', async () => {
    stubSearch(200, { results: [RESULT], timings: {} });
    const { rerender, onOpenSource, onAuthFailure } = setup();
    await search('leave');
    await screen.findByTestId('search-result');

    rerender(
      <SearchPanel
        filenames={[]}
        tags={[]}
        resetKey={1}
        onOpenSource={onOpenSource}
        onAuthFailure={onAuthFailure}
      />,
    );

    expect(screen.queryByTestId('search-result')).not.toBeInTheDocument();
    expect(screen.getByTestId('search-input')).toHaveValue('');
  });
});
