import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useBackupActions } from '../../src/hooks/useBackupActions';

const { api, downloadFile } = vi.hoisted(() => ({
  api: { exportCorpus: vi.fn(), importBackup: vi.fn() },
  downloadFile: vi.fn(),
}));

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>(
    '../../src/api/client',
  );
  return { ...actual, api };
});

vi.mock('../../src/utils/exportChat', async () => {
  const actual = await vi.importActual<typeof import('../../src/utils/exportChat')>(
    '../../src/utils/exportChat',
  );
  return { ...actual, downloadFile };
});

function setup() {
  const pushToast = vi.fn(() => 'preparing-id');
  const dismissToast = vi.fn();
  const hook = renderHook(() =>
    useBackupActions({
      pushToast,
      dismissToast,
      noteAuthFailure: () => false,
      onRestoreQueued: vi.fn(),
    }),
  );
  return { pushToast, dismissToast, hook };
}

afterEach(() => vi.clearAllMocks());

describe('useBackupActions', () => {
  it('keeps "Preparing backup" up until the download starts, then removes it', async () => {
    let resolve: (value: { blob: Blob; filename: string }) => void = () => {};
    api.exportCorpus.mockReturnValue(new Promise((done) => (resolve = done)));
    const { pushToast, dismissToast, hook } = setup();

    act(() => hook.result.current.downloadBackup());
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Preparing backup…', sticky: true }),
    );
    expect(dismissToast).not.toHaveBeenCalled();

    await act(async () => resolve({ blob: new Blob(['PK']), filename: 'docrag-export.zip' }));
    await waitFor(() => expect(dismissToast).toHaveBeenCalledWith('preparing-id'));
    expect(downloadFile).toHaveBeenCalledWith('docrag-export.zip', 'application/zip', expect.any(Blob));
  });

  it('removes it when the backup fails, alongside the failure', async () => {
    api.exportCorpus.mockRejectedValue(new Error('boom'));
    const { pushToast, dismissToast, hook } = setup();

    act(() => hook.result.current.downloadBackup());
    await waitFor(() => expect(dismissToast).toHaveBeenCalledWith('preparing-id'));
    expect(pushToast).toHaveBeenLastCalledWith(expect.objectContaining({ title: 'Backup failed' }));
  });
});
