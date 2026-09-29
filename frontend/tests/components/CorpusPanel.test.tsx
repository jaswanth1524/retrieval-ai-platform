import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import CorpusPanel, { type UploadItem } from '../../src/components/CorpusPanel';

function makeFile(name: string, sizeBytes: number): File {
  const file = new File(['x'], name, { type: 'text/plain' });
  Object.defineProperty(file, 'size', { value: sizeBytes });
  return file;
}

function dataTransferWith(files: File[]): DataTransfer {
  return { files } as unknown as DataTransfer;
}

function setup(overrides: Partial<Parameters<typeof CorpusPanel>[0]> = {}) {
  const props = {
    filenames: [] as string[],
    chunkCounts: {} as Record<string, number>,
    uploads: [] as UploadItem[],
    onUpload: vi.fn().mockResolvedValue(undefined),
    onDelete: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
  render(<CorpusPanel {...props} />);
  return props;
}

describe('CorpusPanel', () => {
  it('shows a re-upload of an indexed document as that document re-indexing', () => {
    // A re-upload's name is already indexed, and in-flight cards used to be filtered
    // out by name — so replacing a document showed no progress at all.
    setup({
      filenames: ['a.txt'],
      chunkCounts: { 'a.txt': 3 },
      uploads: [
        {
          id: 'u1',
          filename: 'a.txt',
          status: 'uploading',
          progress: { state: 'embedding', chunksDone: 2, chunksTotal: 4 },
        },
      ],
    });

    const items = screen.getAllByTestId('corpus-panel-item');
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent('re-indexing 50%');
  });

  it('focuses Cancel on a delete confirm, and Escape backs out of it', async () => {
    setup({ filenames: ['a.txt'], chunkCounts: { 'a.txt': 3 } });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete a.txt')).toBeInTheDocument();
  });

  it('exposes indexing progress as a progressbar', () => {
    setup({
      uploads: [
        {
          id: 'u1',
          filename: 'b.txt',
          status: 'uploading',
          progress: { state: 'embedding', chunksDone: 1, chunksTotal: 4 },
        },
      ],
    });
    expect(screen.getByRole('progressbar', { name: 'Indexing b.txt' })).toHaveAttribute(
      'aria-valuenow',
      '25',
    );
  });

  it('offers re-index on documents with a stored original and flags stale ones', async () => {
    const onReindex = vi.fn();
    setup({
      filenames: ['old.md', 'new.md', 'no-original.md'],
      chunkCounts: { 'old.md': 2, 'new.md': 2, 'no-original.md': 2 },
      staleFilenames: ['old.md', 'no-original.md'],
      reindexableFilenames: ['old.md', 'new.md'],
      onReindex,
    });

    const [oldCard, newCard, noOriginal] = screen.getAllByTestId('corpus-panel-item');
    expect(oldCard).toHaveTextContent('older chunking');
    expect(newCard).not.toHaveTextContent('older chunking');
    // Stale but nothing to re-index from: neither a label the user can't clear nor a
    // button that would 404.
    expect(noOriginal).not.toHaveTextContent('older chunking');
    expect(screen.queryByRole('button', { name: 'Re-index no-original.md' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Re-index old.md' }));
    expect(onReindex).toHaveBeenCalledWith('old.md');
    expect(screen.getByRole('button', { name: 'Re-index new.md' })).toBeInTheDocument();
  });

  it('offers a single bulk re-index once more than one document is stale', async () => {
    const onReindexAllStale = vi.fn();
    const props = {
      filenames: ['a.md', 'b.md'],
      chunkCounts: { 'a.md': 1, 'b.md': 1 },
      reindexableFilenames: ['a.md', 'b.md'],
      onReindex: vi.fn(),
      onReindexAllStale,
    };
    const { unmount } = render(<CorpusPanel {...props} uploads={[]} onUpload={vi.fn()} onDelete={vi.fn()} staleFilenames={['a.md']} />);
    // One stale document: its own ↻ is enough.
    expect(screen.queryByTestId('reindex-all-stale')).not.toBeInTheDocument();
    unmount();

    render(<CorpusPanel {...props} uploads={[]} onUpload={vi.fn()} onDelete={vi.fn()} staleFilenames={['a.md', 'b.md']} />);
    await userEvent.click(screen.getByTestId('reindex-all-stale'));
    expect(onReindexAllStale).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('reindex-all-stale')).toHaveTextContent('Re-index 2 documents');
  });

  it('hides re-index and the stale label entirely when the server keeps no originals', () => {
    setup({
      filenames: ['a.md'],
      chunkCounts: { 'a.md': 1 },
      staleFilenames: ['a.md'],
      reindexableFilenames: ['a.md'],
    });
    expect(screen.queryByRole('button', { name: 'Re-index a.md' })).not.toBeInTheDocument();
    expect(screen.getByTestId('corpus-panel-item')).not.toHaveTextContent('older chunking');
  });

  it('lets a failed upload be retried or dismissed', async () => {
    const onRetryUpload = vi.fn();
    const onDismissUpload = vi.fn();
    setup({
      filenames: ['a.txt'],
      chunkCounts: { 'a.txt': 3 },
      uploads: [
        {
          id: 'u1',
          filename: 'a.txt',
          status: 'error',
          error: 'Indexing failed.',
          file: makeFile('a.txt', 10),
        },
        { id: 'u2', filename: 'b.txt', status: 'error', error: 'Rejected.' },
      ],
      onRetryUpload,
      onDismissUpload,
    });

    // A failed re-upload leaves the previous version indexed: both cards show.
    expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(3);
    expect(screen.getByLabelText('Delete a.txt')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Retry a.txt' }));
    expect(onRetryUpload).toHaveBeenCalledWith('u1');
    // No file kept (e.g. restored from an older state): dismiss only.
    expect(screen.queryByRole('button', { name: 'Retry b.txt' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Dismiss b.txt' }));
    expect(onDismissUpload).toHaveBeenCalledWith('u2');
  });

  it('renders the dropzone with no cards when there are no indexed documents', () => {
    setup();
    expect(screen.getByTestId('upload-dropzone')).toBeInTheDocument();
    expect(screen.queryAllByTestId('corpus-panel-item')).toHaveLength(0);
  });

  it('lists indexed filenames with their chunk counts', () => {
    setup({ filenames: ['a.txt', 'b.pdf'], chunkCounts: { 'a.txt': 3, 'b.pdf': 61 } });

    const items = screen.getAllByTestId('corpus-panel-item');
    expect(items[0]).toHaveTextContent('a.txt');
    expect(items[0]).toHaveTextContent('3 chunks');
    expect(items[1]).toHaveTextContent('b.pdf');
    expect(items[1]).toHaveTextContent('61 chunks');
  });

  it('appends page count, size, and upload time to the detail line when present', () => {
    setup({
      filenames: ['a.txt'],
      chunkCounts: { 'a.txt': 3 },
      pageCounts: { 'a.txt': 2 },
      byteSizes: { 'a.txt': 1024 * 1024 },
      uploadedAts: { 'a.txt': Date.now() / 1000 },
    });

    const item = screen.getByTestId('corpus-panel-item');
    expect(item).toHaveTextContent('3 chunks');
    expect(item).toHaveTextContent('2 pages');
    expect(item).toHaveTextContent('1.0 MB');
    expect(item).toHaveTextContent('just now');
  });

  it('omits page/size/upload-time from the detail line for a filename missing that metadata', () => {
    // The pre-existing case: a filename ingested before this metadata was stamped
    // has none of these fields in the response at all.
    setup({ filenames: ['legacy.txt'], chunkCounts: { 'legacy.txt': 1 } });

    const item = screen.getByTestId('corpus-panel-item');
    expect(item).toHaveTextContent('1 chunks');
    expect(item.textContent).not.toContain('page');
    expect(item.textContent).not.toContain('MB');
  });

  it('renders no search box when there are no documents', () => {
    setup();
    expect(screen.queryByTestId('corpus-panel-search')).not.toBeInTheDocument();
  });

  it('filters the document list by a substring match, case-insensitively', async () => {
    setup({
      filenames: ['contract.pdf', 'invoice.txt', 'README.md'],
      chunkCounts: { 'contract.pdf': 3, 'invoice.txt': 1, 'README.md': 2 },
    });

    await userEvent.type(screen.getByTestId('corpus-panel-search'), 'read');

    const items = screen.getAllByTestId('corpus-panel-item');
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent('README.md');
  });

  it('shows a no-match message and no cards when the filter matches nothing', async () => {
    setup({ filenames: ['a.txt'], chunkCounts: { 'a.txt': 1 } });

    await userEvent.type(screen.getByTestId('corpus-panel-search'), 'zzz');

    expect(screen.queryAllByTestId('corpus-panel-item')).toHaveLength(0);
    expect(screen.getByText('No documents match "zzz".')).toBeInTheDocument();
  });

  it('clicking delete shows an inline confirm, and confirming calls onDelete once', async () => {
    const props = setup({ filenames: ['a.txt'], chunkCounts: { 'a.txt': 1 } });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    expect(screen.getByText('Delete?')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Confirm'));

    expect(props.onDelete).toHaveBeenCalledTimes(1);
    expect(props.onDelete).toHaveBeenCalledWith('a.txt');
  });

  it('clicking cancel dismisses the confirm without calling onDelete', async () => {
    const props = setup({ filenames: ['a.txt'], chunkCounts: { 'a.txt': 1 } });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    await userEvent.click(screen.getByText('Cancel'));

    expect(props.onDelete).not.toHaveBeenCalled();
    expect(screen.queryByText('Delete?')).not.toBeInTheDocument();
  });

  it('shows an error message when onDelete rejects', async () => {
    setup({
      filenames: ['a.txt'],
      chunkCounts: { 'a.txt': 1 },
      onDelete: vi.fn().mockRejectedValue(new Error('No indexed document named a.txt.')),
    });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    await userEvent.click(screen.getByText('Confirm'));

    expect(await screen.findByRole('alert')).toHaveTextContent('No indexed document named a.txt.');
  });

  it('disables delete buttons when disabled is true', () => {
    setup({ filenames: ['a.txt'], chunkCounts: { 'a.txt': 1 }, disabled: true });
    expect(screen.getByLabelText('Delete a.txt')).toBeDisabled();
  });

  it('rejects a file over the configured limit without calling onUpload', async () => {
    const props = setup({ maxUploadBytes: 10 });

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('big.txt', 100));

    expect(screen.getByTestId('upload-size-error')).toHaveTextContent(/too large/);
    expect(screen.queryByTestId('upload-button')).not.toBeInTheDocument();
    expect(props.onUpload).not.toHaveBeenCalled();
  });

  it('stages a file at or under the configured limit and uploads it as an array', async () => {
    const props = setup({ maxUploadBytes: 1000 });

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('small.txt', 10));

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-staged-item')).toHaveTextContent('small.txt');

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(props.onUpload).toHaveBeenCalledTimes(1);
    expect(props.onUpload.mock.calls[0][0]).toHaveLength(1);
    expect(props.onUpload.mock.calls[0][0][0].name).toBe('small.txt');
  });

  it('stages multiple files from one multi-select and uploads all of them', async () => {
    const props = setup({ maxUploadBytes: 1000 });

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('a.txt', 10), makeFile('b.txt', 10)]);

    expect(screen.getAllByTestId('upload-staged-item')).toHaveLength(2);

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(props.onUpload).toHaveBeenCalledTimes(1);
    const uploaded = props.onUpload.mock.calls[0][0] as File[];
    expect(uploaded.map((f) => f.name)).toEqual(['a.txt', 'b.txt']);
  });

  it('removes a staged file via its remove button before upload', async () => {
    const props = setup({ maxUploadBytes: 1000 });

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('a.txt', 10), makeFile('b.txt', 10)]);

    const removeButtons = screen.getAllByTestId('upload-staged-remove');
    await userEvent.click(removeButtons[0]);

    expect(screen.getAllByTestId('upload-staged-item')).toHaveLength(1);

    await userEvent.click(screen.getByTestId('upload-button'));

    const uploaded = props.onUpload.mock.calls[0][0] as File[];
    expect(uploaded.map((f) => f.name)).toEqual(['b.txt']);
  });

  it('accepts valid files dropped onto the drop-zone', () => {
    setup({ maxUploadBytes: 1000 });

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith([makeFile('dropped.txt', 10)]) });

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-button')).toBeInTheDocument();
  });

  it('rejects an oversized file dropped onto the drop-zone', () => {
    setup({ maxUploadBytes: 10 });

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith([makeFile('big.txt', 100)]) });

    expect(screen.getByTestId('upload-size-error')).toHaveTextContent(/too large/);
    expect(screen.queryByTestId('upload-button')).not.toBeInTheDocument();
  });

  it('highlights the drop-zone on dragover and clears it on dragleave', () => {
    setup();

    const dropzone = screen.getByTestId('upload-dropzone');
    expect(dropzone.className).not.toMatch(/--active/);

    fireEvent.dragOver(dropzone);
    expect(dropzone.className).toMatch(/--active/);

    fireEvent.dragLeave(dropzone);
    expect(dropzone.className).not.toMatch(/--active/);
  });

  it('renders a per-file progress card while a job is uploading', () => {
    const uploads: UploadItem[] = [
      {
        id: '1',
        filename: 'a.txt',
        status: 'uploading',
        progress: { state: 'embedding', chunksDone: 2, chunksTotal: 4 },
      },
    ];
    setup({ uploads });

    const items = screen.getAllByTestId('corpus-panel-item');
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent('indexing 50%');
  });

  it('renders a failed upload independently of a successful one', () => {
    const uploads: UploadItem[] = [
      { id: '1', filename: 'a.txt', status: 'success' },
      { id: '2', filename: 'b.txt', status: 'error', error: 'Ingestion failed.' },
    ];
    setup({ filenames: ['a.txt'], chunkCounts: { 'a.txt': 3 }, uploads });

    const items = screen.getAllByTestId('corpus-panel-item');
    // a.txt is both an indexed filename and a 'success' upload entry — only one card.
    expect(items).toHaveLength(2);
    expect(items.some((item) => item.textContent?.includes('a.txt') && item.textContent.includes('3 chunks'))).toBe(
      true,
    );
    expect(items.some((item) => item.textContent?.includes('b.txt') && item.textContent.includes('failed'))).toBe(
      true,
    );
  });
});
