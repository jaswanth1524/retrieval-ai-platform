import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import UploadPanel, { type UploadState } from '../../src/components/UploadPanel';

function makeFile(name: string, sizeBytes: number): File {
  const file = new File(['x'], name, { type: 'text/plain' });
  Object.defineProperty(file, 'size', { value: sizeBytes });
  return file;
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
});
