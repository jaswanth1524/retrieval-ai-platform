import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getDocumentContent: vi.fn() }));
vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>(
    '../../src/api/client',
  );
  return { ...actual, api };
});

import {
  CITATION_WINDOW_RADIUS,
  clearPrefetchedContent,
  prefetchCitationContent,
  takePrefetchedContent,
} from '../../src/utils/contentPrefetch';

const page = { filename: 'a.md', chunks: [], total_chunks: 0, target_found: true };

describe('citation content prefetch', () => {
  beforeEach(() => {
    clearPrefetchedContent();
    api.getDocumentContent.mockReset();
    api.getDocumentContent.mockResolvedValue(page);
  });
  afterEach(() => vi.useRealTimers());

  it('fetches the viewer’s first page once per citation and hands it over once', async () => {
    prefetchCitationContent('a.md', 'c1');
    prefetchCitationContent('a.md', 'c1'); // hovering again doesn't refetch

    expect(api.getDocumentContent).toHaveBeenCalledTimes(1);
    expect(api.getDocumentContent).toHaveBeenCalledWith('a.md', undefined, {
      around: 'c1',
      radius: CITATION_WINDOW_RADIUS,
    });
    await expect(takePrefetchedContent('a.md', 'c1')).resolves.toBe(page);
    expect(takePrefetchedContent('a.md', 'c1')).toBeUndefined();
    expect(takePrefetchedContent('a.md', 'c2')).toBeUndefined();
  });

  it('does not hand over a prefetch older than 30 seconds', () => {
    vi.useFakeTimers();
    prefetchCitationContent('a.md', 'c1');
    vi.advanceTimersByTime(30_000);
    expect(takePrefetchedContent('a.md', 'c1')).toBeUndefined();
  });

  it('forgets a prefetch that failed, so the viewer asks again', async () => {
    api.getDocumentContent.mockRejectedValueOnce(new Error('offline'));
    prefetchCitationContent('a.md', 'c1');
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(takePrefetchedContent('a.md', 'c1')).toBeUndefined();
  });
});
