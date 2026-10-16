import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiErrorMessage, getWorkspace, setWorkspace } from '../api/client';
import type { ToastInput } from './useToasts';
import { useStableHandler } from './useStableHandler';

interface WorkspaceOptions {
  /** WORKSPACES_ENABLED, from /config. */
  enabled: boolean;
  /** The active conversation's workspace, or null while it has no questions. */
  conversationWorkspace: string | null;
  activeConversationId: string | null;
  newConversation: () => void;
  refreshDocuments: () => Promise<void>;
  noteAuthFailure: (err: unknown) => boolean;
  pushToast: (toast: ToastInput) => unknown;
  /** After the workspace in use changed (e.g. close a viewer showing the old one's file). */
  onSwitched?: () => void;
}

/**
 * The workspace in use ('' = the default) and the others the server has.
 *
 * Every request carries the workspace in use (api/client authHeaders). A conversation
 * belongs to the workspace its questions went to: choosing another workspace leaves it
 * (a fresh chat opens), and opening it brings its workspace back — or, when that
 * workspace was deleted since, says so once instead.
 */
export function useWorkspaces({
  enabled,
  conversationWorkspace,
  activeConversationId,
  newConversation,
  refreshDocuments,
  noteAuthFailure,
  pushToast,
  onSwitched,
}: WorkspaceOptions) {
  const [workspace, setWorkspaceState] = useState(getWorkspace);
  const [workspaces, setWorkspaces] = useState<string[]>([]);
  // Whether `workspaces` came from the server yet: until then a conversation's workspace
  // can't be told apart from a deleted one.
  const [loaded, setLoaded] = useState(false);

  const loadWorkspaces = useStableHandler(async () => {
    try {
      const result = await api.listWorkspaces();
      setWorkspaces(result.workspaces);
      setLoaded(true);
    } catch (err) {
      noteAuthFailure(err);
    }
  });
  useEffect(() => {
    if (enabled) void loadWorkspaces();
  }, [enabled, loadWorkspaces]);

  /** A workspace remembered in this browser may have been deleted (or the feature
   *  turned off) since: back to the default before anything is listed from it. */
  const dropRememberedIfGone = useStableHandler(async (signal: AbortSignal) => {
    if (!getWorkspace()) return;
    try {
      const spaces = await api.listWorkspaces(signal);
      if (!spaces.enabled || !spaces.workspaces.includes(getWorkspace())) {
        setWorkspace('');
        setWorkspaceState('');
      }
    } catch {
      // Unreachable or a 401: the requests that follow report it.
    }
  });

  // The corpus is listed again from the new workspace, and scopes naming documents it
  // lacks are pruned.
  const switchWorkspace = useStableHandler((name: string) => {
    setWorkspace(name);
    setWorkspaceState(name);
    onSwitched?.();
    void refreshDocuments().catch((err) => noteAuthFailure(err));
  });

  const chooseWorkspace = useCallback(
    (name: string) => {
      if (conversationWorkspace !== null && conversationWorkspace !== name) newConversation();
      switchWorkspace(name);
    },
    [conversationWorkspace, newConversation, switchWorkspace],
  );

  const missingNoted = useRef<string | null>(null);
  const notify = useStableHandler(pushToast);
  useEffect(() => {
    if (!enabled || !loaded || conversationWorkspace === null) return;
    if (conversationWorkspace === workspace) return;
    if (conversationWorkspace === '' || workspaces.includes(conversationWorkspace)) {
      switchWorkspace(conversationWorkspace);
      return;
    }
    // Deleted since. Questions here now go to the workspace in use; say so once.
    if (missingNoted.current === activeConversationId) return;
    missingNoted.current = activeConversationId;
    notify({
      tone: 'warn',
      title: 'Workspace no longer exists',
      body: `This conversation was in workspace ${conversationWorkspace}; new questions go to ${workspace || 'the default workspace'}.`,
    });
  }, [
    enabled,
    loaded,
    conversationWorkspace,
    workspace,
    workspaces,
    activeConversationId,
    switchWorkspace,
    notify,
  ]);

  const createWorkspace = useCallback(
    async (name: string) => {
      try {
        await api.createWorkspace(name);
      } catch (err) {
        throw new Error(apiErrorMessage(err, 'The workspace was not created.'));
      }
      await loadWorkspaces();
      chooseWorkspace(name);
    },
    [loadWorkspaces, chooseWorkspace],
  );

  // Only an empty workspace can be deleted (the server says 409 otherwise).
  const deleteWorkspace = useCallback(
    async (name: string) => {
      try {
        await api.deleteWorkspace(name);
      } catch (err) {
        throw new Error(apiErrorMessage(err, 'The workspace was not deleted.'));
      }
      chooseWorkspace('');
      await loadWorkspaces();
    },
    [loadWorkspaces, chooseWorkspace],
  );

  return {
    workspace,
    workspaces,
    dropRememberedIfGone,
    chooseWorkspace,
    createWorkspace,
    deleteWorkspace,
  };
}
