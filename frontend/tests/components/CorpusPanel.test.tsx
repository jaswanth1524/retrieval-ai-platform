import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CorpusPanel, { type DocumentMeta, type UploadItem } from '../../src/components/CorpusPanel';

function makeFile(name: string, sizeBytes: number): File {
  const file = new File(['x'], name, { type: 'text/plain' });
  Object.defineProperty(file, 'size', { value: sizeBytes });
  return file;
}

function dataTransferWith(files: File[]): DataTransfer {
  return { files } as unknown as DataTransfer;
}

function meta(overrides: Partial<DocumentMeta> = {}): DocumentMeta {
  return { chunks: 1, stale: false, reindexable: false, tags: [], ...overrides };
}

function setup(overrides: Partial<Parameters<typeof CorpusPanel>[0]> = {}) {
  const props = {
    filenames: [] as string[],
    documents: {} as Record<string, DocumentMeta>,
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
      documents: { 'a.txt': meta({ chunks: 3 }) },
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
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta({ chunks: 3 }) } });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete a.txt')).toBeInTheDocument();
  });

  it('offers Cancel on a card that is indexing, and only there', async () => {
    const onCancelUpload = vi.fn();
    setup({
      filenames: ['a.txt'],
      documents: { 'a.txt': meta() },
      uploads: [
        { id: 'u1', filename: 'b.txt', status: 'uploading' },
        { id: 'u2', filename: 'c.txt', status: 'error', error: 'Ingestion failed.' },
      ],
      onCancelUpload,
    });

    await userEvent.click(screen.getByRole('button', { name: 'Cancel indexing b.txt' }));

    expect(onCancelUpload).toHaveBeenCalledWith('u1');
    // Indexed and failed cards have nothing to stop.
    expect(screen.getAllByTestId('corpus-cancel-upload')).toHaveLength(1);
  });

  it('shows a cancel in progress and does not offer it twice', () => {
    setup({
      uploads: [{ id: 'u1', filename: 'b.txt', status: 'uploading', cancelling: true }],
      onCancelUpload: vi.fn(),
    });

    expect(screen.getByRole('button', { name: 'Cancel indexing b.txt' })).toBeDisabled();
    expect(screen.getByTestId('corpus-panel-item')).toHaveTextContent('cancelling…');
  });

  it('offers no Cancel to a read-only key', () => {
    setup({
      uploads: [{ id: 'u1', filename: 'b.txt', status: 'uploading' }],
      onCancelUpload: vi.fn(),
      readOnly: true,
    });

    expect(screen.queryByTestId('corpus-cancel-upload')).not.toBeInTheDocument();
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

  it('keeps upload available while a re-index runs, but not while a file is sending', () => {
    const reindexing: UploadItem = { id: 'r', filename: 'a.md', status: 'uploading', reindex: true };
    setup({ filenames: ['a.md'], documents: { 'a.md': meta() }, uploads: [reindexing] });
    expect(screen.getByTestId('upload-input')).not.toBeDisabled();

    const sending: UploadItem = {
      id: 'u',
      filename: 'b.md',
      status: 'uploading',
      file: makeFile('b.md', 10),
    };
    setup({ uploads: [sending] });
    expect(screen.getAllByTestId('upload-input').at(-1)).toBeDisabled();
  });

  it('offers re-index on documents with a stored original and flags stale ones', async () => {
    const onReindex = vi.fn();
    setup({
      filenames: ['old.md', 'new.md', 'no-original.md'],
      documents: {
        'old.md': meta({ chunks: 2, stale: true, reindexable: true }),
        'new.md': meta({ chunks: 2, reindexable: true }),
        'no-original.md': meta({ chunks: 2, stale: true }),
      },
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
      onReindex: vi.fn(),
      onReindexAllStale,
    };
    const { unmount } = render(
      <CorpusPanel
        {...props}
        documents={{ 'a.md': meta({ stale: true, reindexable: true }), 'b.md': meta({ reindexable: true }) }}
        uploads={[]}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    // One stale document: its own ↻ is enough.
    expect(screen.queryByTestId('reindex-all-stale')).not.toBeInTheDocument();
    unmount();

    render(
      <CorpusPanel
        {...props}
        documents={{
          'a.md': meta({ stale: true, reindexable: true }),
          'b.md': meta({ stale: true, reindexable: true }),
        }}
        uploads={[]}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId('reindex-all-stale'));
    expect(onReindexAllStale).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('reindex-all-stale')).toHaveTextContent('Re-index 2 documents');
  });

  it('hides re-index and the stale label entirely when the server keeps no originals', () => {
    setup({
      filenames: ['a.md'],
      documents: { 'a.md': meta({ stale: true, reindexable: true }) },
    });
    expect(screen.queryByRole('button', { name: 'Re-index a.md' })).not.toBeInTheDocument();
    expect(screen.getByTestId('corpus-panel-item')).not.toHaveTextContent('older chunking');
  });

  it('lets a failed upload be retried or dismissed', async () => {
    const onRetryUpload = vi.fn();
    const onDismissUpload = vi.fn();
    setup({
      filenames: ['a.txt'],
      documents: { 'a.txt': meta({ chunks: 3 }) },
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
    setup({ filenames: ['a.txt', 'b.pdf'], documents: { 'a.txt': meta({ chunks: 3 }), 'b.pdf': meta({ chunks: 61 }) } });

    const items = screen.getAllByTestId('corpus-panel-item');
    expect(items[0]).toHaveTextContent('a.txt');
    expect(items[0]).toHaveTextContent('3 chunks');
    expect(items[1]).toHaveTextContent('b.pdf');
    expect(items[1]).toHaveTextContent('61 chunks');
  });

  it('appends page count, size, and upload time to the detail line when present', () => {
    setup({
      filenames: ['a.txt'],
      documents: {
        'a.txt': meta({ chunks: 3, pages: 2, bytes: 1024 * 1024, uploadedAt: Date.now() / 1000 }),
      },
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
    setup({ filenames: ['legacy.txt'], documents: { 'legacy.txt': meta() } });

    const item = screen.getByTestId('corpus-panel-item');
    expect(item).toHaveTextContent('1 chunk');
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
      documents: { 'contract.pdf': meta({ chunks: 3 }), 'invoice.txt': meta(), 'README.md': meta({ chunks: 2 }) },
    });

    await userEvent.type(screen.getByTestId('corpus-panel-search'), 'read');

    const items = screen.getAllByTestId('corpus-panel-item');
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent('README.md');
  });

  it('shows a no-match message and no cards when the filter matches nothing', async () => {
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta() } });

    await userEvent.type(screen.getByTestId('corpus-panel-search'), 'zzz');

    expect(screen.queryAllByTestId('corpus-panel-item')).toHaveLength(0);
    expect(screen.getByText('No documents match "zzz".')).toBeInTheDocument();
  });

  it('clicking delete shows an inline confirm, and confirming calls onDelete once', async () => {
    const props = setup({ filenames: ['a.txt'], documents: { 'a.txt': meta() } });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    expect(screen.getByText('Delete?')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Confirm'));

    expect(props.onDelete).toHaveBeenCalledTimes(1);
    expect(props.onDelete).toHaveBeenCalledWith('a.txt');
  });

  it('clicking cancel dismisses the confirm without calling onDelete', async () => {
    const props = setup({ filenames: ['a.txt'], documents: { 'a.txt': meta() } });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    await userEvent.click(screen.getByText('Cancel'));

    expect(props.onDelete).not.toHaveBeenCalled();
    expect(screen.queryByText('Delete?')).not.toBeInTheDocument();
  });

  it('shows an error message when onDelete rejects', async () => {
    setup({
      filenames: ['a.txt'],
      documents: { 'a.txt': meta() },
      onDelete: vi.fn().mockRejectedValue(new Error('No indexed document named a.txt.')),
    });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    await userEvent.click(screen.getByText('Confirm'));

    expect(await screen.findByRole('alert')).toHaveTextContent('No indexed document named a.txt.');
  });

  it('disables delete buttons when disabled is true', () => {
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta() }, disabled: true });
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
    expect(vi.mocked(props.onUpload).mock.calls[0][0]).toHaveLength(1);
    expect(vi.mocked(props.onUpload).mock.calls[0][0][0].name).toBe('small.txt');
  });

  it('clears the file input after a pick, so the same file can be picked again', async () => {
    setup({ maxUploadBytes: 1000 });

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('small.txt', 10));
    expect(input.value).toBe('');
  });

  it('stages multiple files from one multi-select and uploads all of them', async () => {
    const props = setup({ maxUploadBytes: 1000 });

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('a.txt', 10), makeFile('b.txt', 10)]);

    expect(screen.getAllByTestId('upload-staged-item')).toHaveLength(2);

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(props.onUpload).toHaveBeenCalledTimes(1);
    const uploaded = vi.mocked(props.onUpload).mock.calls[0][0] as File[];
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

    const uploaded = vi.mocked(props.onUpload).mock.calls[0][0] as File[];
    expect(uploaded.map((f) => f.name)).toEqual(['b.txt']);
  });

  it('rejects an unsupported type dropped onto the drop-zone', () => {
    setup({ maxUploadBytes: 1000 });

    fireEvent.drop(screen.getByTestId('upload-dropzone'), {
      dataTransfer: dataTransferWith([makeFile('photo.png', 10), makeFile('notes.csv', 10)]),
    });

    expect(screen.getByText(/photo\.png isn't a supported type/)).toBeInTheDocument();
    expect(screen.getByText('notes.csv')).toBeInTheDocument();
  });

  it('opens the delete confirm on one card only when a failed upload shares its name', async () => {
    setup({
      filenames: ['a.txt'],
      documents: { 'a.txt': meta({ chunks: 3 }) },
      uploads: [{ id: 'u1', filename: 'a.txt', status: 'error', error: 'Indexing failed.' }],
    });

    await userEvent.click(screen.getByLabelText('Delete a.txt'));

    expect(screen.getAllByRole('button', { name: 'Confirm' })).toHaveLength(1);
  });

  it('edits a document\'s tags inline and saves them as a list', async () => {
    const onSetTags = vi.fn().mockResolvedValue(undefined);
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta({ tags: ['hr'] }) }, onSetTags });

    expect(screen.getByText('#hr')).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText('Edit tags for a.txt'));
    const input = screen.getByTestId('corpus-tags-input');
    await userEvent.clear(input);
    await userEvent.type(input, 'hr, legal , {Enter}');

    expect(onSetTags).toHaveBeenCalledWith('a.txt', ['hr', 'legal']);
  });

  it('saves and cancels tag edits with buttons, not only Enter and Esc', async () => {
    const onSetTags = vi.fn().mockResolvedValue(undefined);
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta({ tags: ['hr'] }) }, onSetTags });

    await userEvent.click(screen.getByLabelText('Edit tags for a.txt'));
    await userEvent.type(screen.getByTestId('corpus-tags-input'), ', legal');
    await userEvent.click(screen.getByTestId('corpus-tags-save'));
    expect(onSetTags).toHaveBeenCalledWith('a.txt', ['hr', 'legal']);

    await userEvent.click(screen.getByLabelText('Edit tags for a.txt'));
    await userEvent.click(screen.getByTestId('corpus-tags-cancel'));
    expect(screen.queryByTestId('corpus-tags-input')).not.toBeInTheDocument();
    expect(onSetTags).toHaveBeenCalledTimes(1);
    // The control that had focus is gone; focus returns to the one that opened it.
    expect(screen.getByLabelText('Edit tags for a.txt')).toHaveFocus();
  });

  it('keeps the tag editor open with the error when saving fails', async () => {
    const onSetTags = vi.fn().mockRejectedValue(new Error('Tags must be 1-40 printable characters.'));
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta() }, onSetTags });

    await userEvent.click(screen.getByLabelText('Edit tags for a.txt'));
    await userEvent.type(screen.getByTestId('corpus-tags-input'), 'x{Enter}');

    expect(await screen.findByRole('alert')).toHaveTextContent('printable characters');
    expect(screen.getByTestId('corpus-tags-input')).toBeInTheDocument();
  });

  it('accepts valid files dropped onto the drop-zone', () => {
    setup({ maxUploadBytes: 1000 });

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith([makeFile('dropped.txt', 10)]) });

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-button')).toBeInTheDocument();
  });

  it('stages files dropped while an upload runs instead of swallowing them', () => {
    setup({
      maxUploadBytes: 1000,
      uploads: [{ id: 'u1', filename: 'b.txt', status: 'uploading', file: makeFile('b.txt', 10) }],
    });

    fireEvent.drop(screen.getByTestId('upload-dropzone'), {
      dataTransfer: dataTransferWith([makeFile('next.txt', 10)]),
    });

    expect(screen.getByText('next.txt')).toBeInTheDocument();
    // Sent once the running upload finishes.
    expect(screen.getByTestId('upload-button')).toBeDisabled();
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
    setup({ filenames: ['a.txt'], documents: { 'a.txt': meta({ chunks: 3 }) }, uploads });

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

  describe('reading, selecting, filtering and sorting documents', () => {
    const THREE = {
      filenames: ['b.md', 'a.md', 'c.md'],
      documents: {
        'b.md': meta({ tags: ['hr'], bytes: 300, uploadedAt: 3 }),
        'a.md': meta({ tags: ['legal'], bytes: 100, uploadedAt: 1, reindexable: true }),
        'c.md': meta({ tags: ['hr'], bytes: 200, uploadedAt: 2, reindexable: true }),
      },
    };
    const names = () =>
      screen.getAllByTestId('corpus-panel-item').map((card) => card.getAttribute('data-corpus-card'));

    afterEach(() => localStorage.clear());

    it('opens a document to read from its card, read-only key included', async () => {
      const onRead = vi.fn();
      setup({ ...THREE, onRead, readOnly: true });

      await userEvent.click(screen.getByRole('button', { name: 'Read a.md' }));

      expect(onRead).toHaveBeenCalledWith('a.md');
      // Nothing that would change the corpus.
      expect(screen.queryByTestId('corpus-select')).not.toBeInTheDocument();
    });

    it('sorts by name, newest or size and remembers the choice', async () => {
      setup(THREE);
      expect(names()).toEqual(['b.md', 'a.md', 'c.md']);

      await userEvent.selectOptions(screen.getByTestId('corpus-sort'), 'name');
      expect(names()).toEqual(['a.md', 'b.md', 'c.md']);
      await userEvent.selectOptions(screen.getByTestId('corpus-sort'), 'newest');
      expect(names()).toEqual(['b.md', 'c.md', 'a.md']);
      await userEvent.selectOptions(screen.getByTestId('corpus-sort'), 'largest');
      expect(names()).toEqual(['b.md', 'c.md', 'a.md']);
      expect(localStorage.getItem('docrag-corpus-sort')).toBe('largest');
    });

    it('filters by tag', async () => {
      setup(THREE);

      await userEvent.selectOptions(screen.getByTestId('corpus-tag-filter'), 'hr');

      expect(names()).toEqual(['b.md', 'c.md']);
    });

    it('deletes the selected documents one at a time and keeps the ones that failed', async () => {
      const onDelete = vi.fn(async (filename: string) => {
        if (filename === 'c.md') throw new Error('nope');
      });
      setup({ ...THREE, onDelete });

      await userEvent.click(screen.getByLabelText('Select a.md'));
      await userEvent.click(screen.getByLabelText('Select c.md'));
      expect(screen.getByTestId('corpus-bulk')).toHaveTextContent('2 selected');
      await userEvent.click(screen.getByTestId('corpus-bulk-delete'));
      await userEvent.click(screen.getByTestId('corpus-bulk-delete-confirm'));

      expect(onDelete.mock.calls.map(([filename]) => filename)).toEqual(['a.md', 'c.md']);
      expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't delete: c.md");
      expect(screen.getByTestId('corpus-bulk')).toHaveTextContent('1 selected');
    });

    it('adds a tag to every selected document that lacks it, and re-indexes those it can', async () => {
      const onSetTags = vi.fn().mockResolvedValue(undefined);
      const onReindex = vi.fn();
      setup({ ...THREE, onSetTags, onReindex });

      await userEvent.click(screen.getByTestId('corpus-select-all'));
      await userEvent.type(screen.getByTestId('corpus-bulk-tag-input'), 'HR');
      await userEvent.click(screen.getByTestId('corpus-bulk-tag'));

      // b.md and c.md already carry "hr" (tags match case-insensitively).
      expect(onSetTags.mock.calls).toEqual([['a.md', ['legal', 'HR']]]);

      await userEvent.click(screen.getByTestId('corpus-bulk-reindex'));
      expect(onReindex.mock.calls.map(([filename]) => filename).sort()).toEqual(['a.md', 'c.md']);
    });

    it('never acts on a selected document the filter now hides', async () => {
      const onDelete = vi.fn().mockResolvedValue(undefined);
      setup({ ...THREE, onDelete });

      await userEvent.click(screen.getByTestId('corpus-select-all'));
      expect(screen.getByTestId('corpus-bulk')).toHaveTextContent('3 selected');
      await userEvent.selectOptions(screen.getByTestId('corpus-tag-filter'), 'hr');
      expect(screen.getByTestId('corpus-bulk')).toHaveTextContent('2 selected');
      await userEvent.selectOptions(screen.getByTestId('corpus-tag-filter'), '');

      // a.md was hidden while filtered: it left the selection, and stays out of it.
      expect(screen.getByTestId('corpus-bulk')).toHaveTextContent('2 selected');
      await userEvent.click(screen.getByTestId('corpus-bulk-delete'));
      await userEvent.click(screen.getByTestId('corpus-bulk-delete-confirm'));
      expect(onDelete.mock.calls.map(([filename]) => filename).sort()).toEqual(['b.md', 'c.md']);
    });

    it('re-indexes the selection in one batch when the parent can', async () => {
      const onReindex = vi.fn();
      const onReindexMany = vi.fn();
      setup({ ...THREE, onReindex, onReindexMany });

      await userEvent.click(screen.getByTestId('corpus-select-all'));
      await userEvent.click(screen.getByTestId('corpus-bulk-reindex'));

      expect(onReindexMany).toHaveBeenCalledTimes(1);
      expect([...onReindexMany.mock.calls[0][0]].sort()).toEqual(['a.md', 'c.md']);
      expect(onReindex).not.toHaveBeenCalled();
    });

    it('puts focus on the filter after a bulk delete empties the selection', async () => {
      function Harness() {
        const [files, setFiles] = useState(['a.md', 'b.md', 'c.md']);
        return (
          <CorpusPanel
            filenames={files}
            documents={{ 'a.md': meta(), 'b.md': meta(), 'c.md': meta() }}
            uploads={[]}
            onUpload={vi.fn()}
            onDelete={async (filename) => setFiles((prev) => prev.filter((name) => name !== filename))}
          />
        );
      }
      render(<Harness />);

      await userEvent.click(screen.getByLabelText('Select a.md'));
      await userEvent.click(screen.getByLabelText('Select b.md'));
      await userEvent.click(screen.getByTestId('corpus-bulk-delete'));
      await userEvent.click(screen.getByTestId('corpus-bulk-delete-confirm'));

      await waitFor(() => expect(screen.queryByTestId('corpus-bulk')).not.toBeInTheDocument());
      expect(document.activeElement).toBe(screen.getByTestId('corpus-panel-search'));
    });

    it('moves focus to the next card after a confirmed delete', async () => {
      function Harness() {
        const [files, setFiles] = useState(['a.md', 'b.md']);
        return (
          <CorpusPanel
            filenames={files}
            documents={{ 'a.md': meta(), 'b.md': meta() }}
            uploads={[]}
            onUpload={vi.fn()}
            onDelete={async (filename) => setFiles((prev) => prev.filter((name) => name !== filename))}
            onRead={vi.fn()}
          />
        );
      }
      render(<Harness />);

      await userEvent.click(screen.getByLabelText('Delete a.md'));
      await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));

      // The next card's first control, not <body>.
      expect(await screen.findByLabelText('Read b.md')).toHaveFocus();
    });
  });
});
