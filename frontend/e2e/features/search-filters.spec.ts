// The Search panel's filters and similar passages.
import { answerStream, expect, GUIDE_SOURCE, test } from '../fixtures';
import { open, ask } from '../helpers';

test('search filters narrow the request and the match is marked', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.searchResults = [
    {
      rank: 1,
      filename: 'guide.md',
      page: 1,
      section: 'Retrieval',
      chunk_id: 'c1',
      chunk_ordinal: 1,
      text: GUIDE_SOURCE.text,
      retrieval_score: 0.03,
      rerank_score: 0.93,
      spans: [{ start: 0, end: 20, quote: 'DocRAG fuses dense and BM25 results' }],
    },
  ];
  await page.goto('/');
  await page.getByTestId('rail-search').click();
  await page.getByText('Filters').click();
  await page.getByTestId('search-filter-section').fill('Retrieval');
  await page.getByTestId('search-filter-page-min').fill('1');
  await page.getByTestId('search-input').fill('fusion');
  await page.getByTestId('search-submit').click();
  await expect(page.getByTestId('search-match')).toHaveText('DocRAG fuses dense and BM25 results');
  expect(api.searches.at(-1)).toMatchObject({ query: 'fusion', section: 'Retrieval', page_min: 1 });
});

test('passages like one in the viewer are searched for in the Search panel', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  await page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` }).click();
  await page
    .getByRole('dialog')
    .getByTestId('viewer-chunk')
    .filter({ hasText: GUIDE_SOURCE.text })
    .getByTestId('viewer-similar')
    .click();

  await expect(page.getByTestId('search-like')).toContainText(`Passages like ${GUIDE_SOURCE.filename}`);
  expect(api.searches.at(-1)).toMatchObject({ query: '', like_chunk_id: GUIDE_SOURCE.chunk_id });
});
