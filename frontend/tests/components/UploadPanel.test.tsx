import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import UploadPanel, { type UploadState } from '../../src/components/UploadPanel';

function makeFile(name: string, sizeBytes: number): File {
  const file = new File(['x'], name, { type: 'text/plain' });
  Object.defineProperty(file, 'size', { value: sizeBytes });
  return file;
}

function dataTransferWith(file: File): DataTransfer {
  return { files: [file] } as unknown as DataTransfer;
}

describe('UploadPanel', () => {
  it('rejects a file over the configured limit without calling onUpload', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} maxUploadBytes={10} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('big.txt', 100));

    expect(screen.getByTestId('upload-size-error')).toHaveTextContent(/too large/);
    expect(screen.getByTestId('upload-button')).toBeDisabled();
    expect(onUpload).not.toHaveBeenCalled();
  });

  it('accepts a file at or under the configured limit', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} maxUploadBytes={1000} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, makeFile('small.txt', 10));

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-button')).toBeEnabled();

    await userEvent.click(screen.getByTestId('upload-button'));

    expect(onUpload).toHaveBeenCalledTimes(1);
  });

  it('falls back to the default limit when maxUploadBytes is not supplied', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} />);

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    // Well under the 50MB default — should be accepted with no config loaded yet.
    await userEvent.upload(input, makeFile('small.txt', 10));

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
  });

  it('notifies the owner when a valid file is selected, so a stale result can be cleared', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const onFileSelected = vi.fn();
    const state: UploadState = {
      status: 'success',
      result: { filename: 'old.txt', sections_parsed: 1, chunks_ingested: 1, collection_name: 'c' },
    };
    render(<UploadPanel onUpload={onUpload} state={state} onFileSelected={onFileSelected} />);

    await userEvent.upload(screen.getByTestId('upload-input'), makeFile('new.txt', 10));

    expect(onFileSelected).toHaveBeenCalledTimes(1);
  });

  it('also notifies the owner when a rejected oversized file is selected (clears a stale prior result)', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const onFileSelected = vi.fn();
    const state: UploadState = { status: 'idle' };
    render(
      <UploadPanel onUpload={onUpload} state={state} maxUploadBytes={10} onFileSelected={onFileSelected} />,
    );

    await userEvent.upload(screen.getByTestId('upload-input'), makeFile('big.txt', 100));

    expect(onFileSelected).toHaveBeenCalledTimes(1);
  });

  it('accepts a valid file dropped onto the drop-zone', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} maxUploadBytes={1000} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith(makeFile('dropped.txt', 10)) });

    expect(screen.queryByTestId('upload-size-error')).not.toBeInTheDocument();
    expect(screen.getByTestId('upload-button')).toBeEnabled();
  });

  it('rejects an oversized file dropped onto the drop-zone', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} maxUploadBytes={10} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith(makeFile('big.txt', 100)) });

    expect(screen.getByTestId('upload-size-error')).toHaveTextContent(/too large/);
    expect(screen.getByTestId('upload-button')).toBeDisabled();
  });

  it('highlights the drop-zone on dragover and clears it on dragleave', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    expect(dropzone.className).not.toMatch(/--active/);

    fireEvent.dragOver(dropzone);
    expect(dropzone.className).toMatch(/--active/);

    fireEvent.dragLeave(dropzone);
    expect(dropzone.className).not.toMatch(/--active/);
  });

  it('clears the active highlight after a drop', () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    const state: UploadState = { status: 'idle' };
    render(<UploadPanel onUpload={onUpload} state={state} maxUploadBytes={1000} />);

    const dropzone = screen.getByTestId('upload-dropzone');
    fireEvent.dragOver(dropzone);
    fireEvent.drop(dropzone, { dataTransfer: dataTransferWith(makeFile('dropped.txt', 10)) });

    expect(dropzone.className).not.toMatch(/--active/);
  });
});
