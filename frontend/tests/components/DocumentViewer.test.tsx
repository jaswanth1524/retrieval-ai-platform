import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DocumentViewer from '../../src/components/DocumentViewer';
import { api } from '../../src/api/client';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return { ...actual, api: { ...actual.api, getDocumentContent: vi.fn() } };
});

const getContentMock = vi.mocked(api.getDocumentContent);

afterEach(() => getContentMock.mockReset());

const CONTENT = {
  filename: 'guide.md',
  chunks: [
    { chunk_id: 'c1', page: 1, section: 'Intro', text: 'First chunk.', chunk_ordinal: 1 },
    { chunk_id: 'c2', page: 1, section: 'Setup', text: 'Second chunk.', chunk_ordinal: 2 },
  ],
};

describe('DocumentViewer', () => {
  it('fetches and renders document chunks, marking the target', async () => {
    getContentMock.mockResolvedValue(CONTENT);

    render(<DocumentViewer filename="guide.md" chunkId="c2" onClose={vi.fn()} />);

    await waitFor(() => expect(screen.getAllByTestId('viewer-chunk')).toHaveLength(2));
    expect(getContentMock).toHaveBeenCalledWith('guide.md', expect.anything());
    const target = screen.getAllByTestId('viewer-chunk').find((c) => c.textContent?.includes('Second chunk.'));
    expect(target?.className).toContain('document-viewer__chunk--target');
  });

  it('closes on the close button and Escape', async () => {
    getContentMock.mockResolvedValue(CONTENT);
    const onClose = vi.fn();

    render(<DocumentViewer filename="guide.md" chunkId={null} onClose={onClose} />);
    await waitFor(() => expect(screen.getAllByTestId('viewer-chunk')).toHaveLength(2));

    await userEvent.click(screen.getByTestId('viewer-close'));
    expect(onClose).toHaveBeenCalled();

    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('shows an error message when the fetch fails', async () => {
    const { ApiClientError } = await vi.importActual<typeof import('../../src/api/client')>(
      '../../src/api/client',
    );
    getContentMock.mockRejectedValue(new ApiClientError('nope'));

    render(<DocumentViewer filename="guide.md" chunkId={null} onClose={vi.fn()} />);

    expect(await screen.findByRole('alert')).toHaveTextContent('nope');
  });

  function makeChunk(i: number): { chunk_id: string; page: number; section: string; text: string; chunk_ordinal: number } {
    return { chunk_id: `c${i}`, page: 1, section: 'Body', text: `Chunk ${i}`, chunk_ordinal: i };
  }

  describe('windowing on a large document', () => {
    const BIG_CONTENT = {
      filename: 'big.md',
      chunks: Array.from({ length: 500 }, (_, i) => makeChunk(i)),
    };

    it('renders far fewer than every chunk, with the deep target still present', async () => {
      // Regression guard: a naive "render the first N" would put a citation deep in a
      // large document (here, chunk 400 of 500) out of reach entirely.
      getContentMock.mockResolvedValue(BIG_CONTENT);

      render(<DocumentViewer filename="big.md" chunkId="c400" onClose={vi.fn()} />);

      await waitFor(() => expect(screen.getAllByTestId('viewer-chunk').length).toBeLessThan(500));
      const rendered = screen.getAllByTestId('viewer-chunk');

      const target = rendered.find((c) => c.textContent?.includes('Chunk 400'));
      expect(target).toBeDefined();
      expect(target?.className).toContain('document-viewer__chunk--target');
    });

    it('expands the window on Load earlier / Load later, and Show all renders everything', async () => {
      getContentMock.mockResolvedValue(BIG_CONTENT);

      render(<DocumentViewer filename="big.md" chunkId="c400" onClose={vi.fn()} />);
      // Wait for the windowed render specifically, not just "some chunks exist" — the
      // fetch resolving triggers an unwindowed render for one tick before the
      // centering effect settles, and the Load-earlier button only appears once
      // windowed.
      await waitFor(() => expect(screen.getByTestId('viewer-load-earlier')).toBeInTheDocument());

      const before = screen.getAllByTestId('viewer-chunk').length;
      await userEvent.click(screen.getByTestId('viewer-load-earlier'));
      expect(screen.getAllByTestId('viewer-chunk').length).toBeGreaterThan(before);

      await userEvent.click(screen.getByTestId('viewer-show-all'));
      expect(screen.getAllByTestId('viewer-chunk')).toHaveLength(500);
      expect(screen.queryByTestId('viewer-load-earlier')).not.toBeInTheDocument();
      expect(screen.queryByTestId('viewer-load-later')).not.toBeInTheDocument();
    });

    it('re-centers the window on a new citation into the same open document', async () => {
      // App.tsx doesn't remount DocumentViewer when the reader clicks a different
      // citation for a document that's already open — only the chunkId prop changes.
      getContentMock.mockResolvedValue(BIG_CONTENT);

      const { rerender } = render(<DocumentViewer filename="big.md" chunkId="c10" onClose={vi.fn()} />);
      // Same reasoning as above: wait for the settled windowed render before asserting
      // Chunk 480 is absent, or this could observe the transient unwindowed render
      // that (briefly) contains every chunk.
      await waitFor(() => expect(screen.getAllByTestId('viewer-chunk').length).toBeLessThan(500));
      expect(screen.queryByText('Chunk 480')).not.toBeInTheDocument();

      rerender(<DocumentViewer filename="big.md" chunkId="c480" onClose={vi.fn()} />);

      await waitFor(() => {
        const target = screen
          .getAllByTestId('viewer-chunk')
          .find((c) => c.textContent?.includes('Chunk 480'));
        expect(target?.className).toContain('document-viewer__chunk--target');
      });
    });
  });
});
