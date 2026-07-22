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
});
