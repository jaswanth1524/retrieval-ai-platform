import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError, api, getWorkspace, onWorkspaceMissing, setWorkspace } from '../../src/api/client';
import type { ApiStatus } from '../../src/api/types';
import { useWorkspaces } from '../../src/hooks/useWorkspaces';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return {
    ...actual,
    onWorkspaceMissing: vi.fn(),
    api: { ...actual.api, listWorkspaces: vi.fn(), createWorkspace: vi.fn(), deleteWorkspace: vi.fn() },
  };
});

const listWorkspaces = vi.mocked(api.listWorkspaces);
const createWorkspace = vi.mocked(api.createWorkspace);

interface Props {
  apiStatus: ApiStatus;
  conversationWorkspace: string | null;
}

function setup(initial: Partial<Props> = {}) {
  const options = {
    relistCorpus: vi.fn(async () => {}),
    noteAuthFailure: vi.fn(() => false),
    pushToast: vi.fn(),
    newConversation: vi.fn(),
    onSwitched: vi.fn(),
  };
  const hook = renderHook(
    (props: Props) =>
      useWorkspaces({ enabled: true, activeConversationId: 'c1', ...options, ...props }),
    { initialProps: { apiStatus: 'ok', conversationWorkspace: null, ...initial } as Props },
  );
  return { ...hook, ...options };
}

describe('useWorkspaces', () => {
  beforeEach(() => {
    listWorkspaces.mockReset();
    createWorkspace.mockReset();
    listWorkspaces.mockResolvedValue({ enabled: true, workspaces: ['legal'] });
  });
  afterEach(() => setWorkspace(''));

  it('reads the list again once a refused key works', async () => {
    const { rerender } = setup();
    await waitFor(() => expect(listWorkspaces).toHaveBeenCalledTimes(1));

    rerender({ apiStatus: 'unauthorized', conversationWorkspace: null });
    rerender({ apiStatus: 'ok', conversationWorkspace: null });
    await waitFor(() => expect(listWorkspaces).toHaveBeenCalledTimes(2));
  });

  it('reads the list again before calling a conversation’s workspace deleted', async () => {
    // Made in another tab after this one read the list.
    listWorkspaces
      .mockResolvedValueOnce({ enabled: true, workspaces: ['legal'] })
      .mockResolvedValue({ enabled: true, workspaces: ['legal', 'research'] });
    const { result, pushToast } = setup({ conversationWorkspace: 'research' });

    await waitFor(() => expect(result.current.workspace).toBe('research'));
    expect(listWorkspaces).toHaveBeenCalledTimes(2);
    expect(pushToast).not.toHaveBeenCalled();
  });

  it('still says a workspace is gone when the second read agrees', async () => {
    const { result, pushToast } = setup({ conversationWorkspace: 'archive' });
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Workspace no longer exists' })),
    );
    expect(listWorkspaces).toHaveBeenCalledTimes(2);
    expect(result.current.workspace).toBe('');
  });

  it('shows a refused create as the key banner or the read-only UI, and says why', async () => {
    const refused = new ApiClientError('Missing or invalid API key.', 401);
    createWorkspace.mockRejectedValue(refused);
    const { result, noteAuthFailure } = setup();

    await expect(act(() => result.current.createWorkspace('notes'))).rejects.toThrow(
      'Missing or invalid API key.',
    );
    expect(noteAuthFailure).toHaveBeenCalledWith(refused);
    expect(getWorkspace()).toBe('');
  });

  it('keeps a created workspace in the list it is chosen from, even if the list can’t be read', async () => {
    createWorkspace.mockResolvedValue({ name: 'notes' });
    const { result } = setup();
    await waitFor(() => expect(result.current.workspaces).toEqual(['legal']));
    listWorkspaces.mockRejectedValue(new ApiClientError('Qdrant is down.', 503));

    await act(() => result.current.createWorkspace('notes'));

    expect(result.current.workspace).toBe('notes');
    expect(result.current.workspaces).toContain('notes');
  });

  it('falls back to the default when the workspace in use is gone', async () => {
    setWorkspace('legal');
    const { result, pushToast, relistCorpus, onSwitched } = setup();
    await waitFor(() => expect(listWorkspaces).toHaveBeenCalledTimes(1));
    const listener = vi.mocked(onWorkspaceMissing).mock.calls.at(-1)?.[0];

    // A request left running in another workspace doesn't move this one.
    act(() => listener?.('research'));
    expect(result.current.workspace).toBe('legal');

    act(() => listener?.('legal'));
    expect(result.current.workspace).toBe('');
    expect(getWorkspace()).toBe('');
    expect(onSwitched).toHaveBeenCalled();
    expect(relistCorpus).toHaveBeenCalled();
    expect(pushToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Workspace no longer exists', id: 'workspace-missing' }),
    );
    await waitFor(() => expect(listWorkspaces).toHaveBeenCalledTimes(2));
  });
});
