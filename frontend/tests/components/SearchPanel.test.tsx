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
    expect(onOpenSource).toHaveBeenCalledWith('policy.pdf', 'c1', undefined);
  });

  it('asks where an expected document was lost and shows each result\'s leg ranks', async () => {
    const fetchMock = stubSearch(200, {
      results: [{ ...RESULT, dense_rank: 2, sparse_rank: null }],
      timings: {},
      explanation: {
        target: 'any passage of handbook.pdf',
        outcome: 'below_min_score',
        detail: 'The cross-encoder scored it 0.050, under the bar of 0.15 (RERANK_MIN_SCORE).',
      },
    });
    setup();
    await userEvent.type(screen.getByTestId('search-expect'), 'handbook.pdf');
    await search('leave');

    expect(await screen.findByTestId('search-explanation')).toHaveTextContent('under the bar');
    expect(screen.getByText(/dense #2 · BM25 #–/)).toBeInTheDocument();
    const body = JSON.parse(String((fetchMock.mock.calls[0] as unknown[])[1] && ((fetchMock.mock.calls[0] as unknown[])[1] as RequestInit).body));
    expect(body).toMatchObject({ explain: true, expect_filename: 'handbook.pdf' });
  });

  it('groups results by document, remembers the search, and moves with j and k', async () => {
    localStorage.clear();
    const results = [
      { ...RESULT, chunk_id: 'a1', filename: 'a.pdf', rank: 1 },
      { ...RESULT, chunk_id: 'b1', filename: 'b.pdf', rank: 2 },
      { ...RESULT, chunk_id: 'a2', filename: 'a.pdf', rank: 3 },
    ];
    stubSearch(200, { results, timings: {} });
    setup();
    await search('leave');
    await screen.findAllByTestId('search-result');
    await userEvent.click(screen.getByTestId('search-group'));
    expect(screen.getAllByTestId('search-group-head').map((h) => h.textContent)).toEqual(['a.pdf', 'b.pdf']);
    expect(JSON.parse(localStorage.getItem('docrag-recent-searches') ?? '[]')).toEqual(['leave']);

    const buttons = screen.getAllByTestId('search-result');
    buttons[0].focus();
    await userEvent.keyboard('j');
    expect(document.activeElement).toBe(buttons[1]);
    await userEvent.keyboard('k');
    expect(document.activeElement).toBe(buttons[0]);
  });

  it('notes when the passages are only the closest ones', async () => {
    const weak = { results: [{ ...RESULT, rerank_score: 0.11 }], timings: {}, weak_matches: true };
    stubSearch(200, weak);
    setup();

    await search('laptop');

    expect(await screen.findByTestId('search-weak')).toHaveTextContent('No strong matches');
    expect(screen.getByTestId('search-status')).toHaveTextContent(
      'No strong matches; 1 closest passage shown.',
    );
    expect(screen.getByTestId('search-result')).toBeInTheDocument();

    stubSearch(200, { results: [RESULT], timings: {}, weak_matches: false });
    await search(' leave');
    await screen.findByText('0.91');
    expect(screen.queryByTestId('search-weak')).not.toBeInTheDocument();
    expect(screen.getByTestId('search-status')).toHaveTextContent('1 passage found.');
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

  it("passes on the rate limit's own message, which says when to try again", async () => {
    stubSearch(429, { detail: 'Too many requests; try again in 12 s.' });
    setup();

    await search('leave');

    expect((await screen.findByTestId('search-error')).textContent).toBe(
      'Too many requests; try again in 12 s.',
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

  it('reads out how many passages a search found', async () => {
    stubSearch(200, { results: [RESULT, { ...RESULT, chunk_id: 'c2', rank: 2 }] });
    setup();
    expect(screen.getByRole('status')).toHaveTextContent('');

    await search('leave');

    expect(await screen.findByRole('status')).toHaveTextContent('2 passages found.');
  });
});
