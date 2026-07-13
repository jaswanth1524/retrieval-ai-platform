import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import UploadPanel, { type UploadItem } from '../../src/components/UploadPanel';

function makeFile(name: string, sizeBytes: number): File {
  const file = new File(['x'], name, { type: 'text/plain' });
  Object.defineProperty(file, 'size', { value: sizeBytes });
  return file;
}

function dataTransferWith(files: File[]): DataTransfer {
  return { files } as unknown as DataTransfer;
}

describe('UploadPanel', () => {
  it('rejects a file over the configured limit without calling onUpload', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={10} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('big.txt', 100));

    expect(screen.getByTestId('upload-size-error')).toHaveTextContent(/too large/);
    expect(screen.getByTestId('upload-button')).toBeDisabled();
    expect(onUpload).not.toHaveBeenCalled();
  });

  it('stages a file at or under the configured limit and uploads it as an array', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={1000} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('small.txt', 10));

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-staged-item')).toHaveTextContent('small.txt');
    expect(screen.getByTestId('upload-button')).toBeEnabled();

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(onUpload).toHaveBeenCalledTimes(1);
    expect(onUpload.mock.calls[0][0]).toHaveLength(1);
    expect(onUpload.mock.calls[0][0][0].name).toBe('small.txt');
  });

  it('falls back to the default limit when maxUploadBytes is not supplied', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    // Well under the 50MB default — should be accepted with no config loaded yet.
    await userEvent.upload(input, makeFile('small.txt', 10));

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
  });

  it('stages multiple files from one multi-select and uploads all of them', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={1000} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('a.txt', 10), makeFile('b.txt', 10)]);

    expect(screen.getAllByTestId('upload-staged-item')).toHaveLength(2);

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(onUpload).toHaveBeenCalledTimes(1);
    const uploaded = onUpload.mock.calls[0][0] as File[];
    expect(uploaded.map((f) => f.name)).toEqual(['a.txt', 'b.txt']);
  });

  it('removes a staged file via its remove button before upload', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={1000} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('a.txt', 10), makeFile('b.txt', 10)]);

    const removeButtons = screen.getAllByTestId('upload-staged-remove');
    await userEvent.click(removeButtons[0]);

    expect(screen.getAllByTestId('upload-staged-item')).toHaveLength(1);

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(onUpload).toHaveBeenCalledTimes(1);
    const uploaded = onUpload.mock.calls[0][0] as File[];
    expect(uploaded.map((f) => f.name)).toEqual(['b.txt']);
  });

  it('rejects an oversized file but keeps the other valid ones staged', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={50} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('ok.txt', 10), makeFile('big.txt', 100)]);

    expect(screen.getAllByTestId('upload-size-error')).toHaveLength(1);
    expect(screen.getAllByTestId('upload-staged-item')).toHaveLength(1);
    expect(screen.getByTestId('upload-button')).toBeEnabled();
  });

  it('accepts valid files dropped onto the drop-zone', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={1000} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith([makeFile('dropped.txt', 10)]) });

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-button')).toBeEnabled();
  });

  it('rejects an oversized file dropped onto the drop-zone', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={10} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith([makeFile('big.txt', 100)]) });

    expect(screen.getByTestId('upload-size-error')).toHaveTextContent(/too large/);
    expect(screen.getByTestId('upload-button')).toBeDisabled();
  });

  it('highlights the drop-zone on dragover and clears it on dragleave', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    expect(dropzone.className).not.toMatch(/--active/);

    fireEvent.dragOver(dropzone);
    expect(dropzone.className).toMatch(/--active/);

    fireEvent.dragLeave(dropzone);
    expect(dropzone.className).not.toMatch(/--active/);
  });

  it('clears the active highlight after a drop', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<UploadPanel onUpload={onUpload} uploads={[]} maxUploadBytes={1000} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.dragOver(dropzone);
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith([makeFile('dropped.txt', 10)]) });

    expect(dropzone.className).not.toMatch(/--active/);
  });

  it('renders a per-file progress row while a job is uploading', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const uploads: UploadItem[] = [
      {
        id: '1',
        filename: 'a.txt',
        status: 'uploading',
        progress: { state: 'embedding', chunksDone: 2, chunksTotal: 4 },
      },
    ];
    render(<UploadPanel onUpload={onUpload} uploads={uploads} />);

    expect(screen.getByTestId('upload-job-item')).toHaveTextContent(/Embedding\.\.\. \(2\/4\)/);
  });

  it('renders per-file success and error rows independently', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const uploads: UploadItem[] = [
      {
        id: '1',
        filename: 'a.txt',
        status: 'success',
        result: { filename: 'a.txt', sections_parsed: 1, chunks_ingested: 3, collection_name: 'docrag_documents' },
      },
      { id: '2', filename: 'b.txt', status: 'error', error: 'Ingestion failed.' },
    ];
    render(<UploadPanel onUpload={onUpload} uploads={uploads} />);

    const items = screen.getAllByTestId('upload-job-item');
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent('a.txt');
    expect(items[0]).toHaveTextContent(/3 chunks/);
    expect(items[1]).toHaveTextContent('b.txt');
    expect(items[1]).toHaveTextContent('Ingestion failed.');
  });
});
