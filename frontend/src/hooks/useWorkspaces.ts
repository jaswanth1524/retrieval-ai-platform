import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiErrorMessage, getWorkspace, onWorkspaceMissing, setWorkspace } from '../api/client';
import type { ApiStatus } from '../api/types';
import type { ToastInput } from './useToasts';
import { useStableHandler } from './useStableHandler';

interface WorkspaceOptions {
  /** WORKSPACES_ENABLED, from /config. */
  enabled: boolean;
  /** The server's reachability and key state: the list is read again once a rejected
   *  key works (it 401'd while it didn't). */
  apiStatus: ApiStatus;
  /** The active conversation's workspace, or null while it has no questions. */
  conversationWorkspace: string | null;
  activeConversationId: string | null;
  newConversation: () => void;
  /** Lists the corpus of the workspace now in use, retrying a failure (useAppBoot). */
  relistCorpus: () => Promise<void>;
  noteAuthFailure: (err: unknown) => boolean;
  pushToast: (toast: ToastInput) => unknown;
  /** After the workspace in use changed, before its corpus is listed: drop what belongs
   *  to the old one (its corpus, a viewer showing its file, its search results). */
  onSwitched?: () => void;
}

// One notice for "the workspace is gone", however many places notice it.
const MISSING_TOAST_ID = 'workspace-missing';

/**
 * The workspace in use ('' = the default) and the others the server has.
 *
 * Every request carries the workspace in use (api/client authHeaders). A conversation
 * belongs to the workspace its questions went to: choosing another workspace leaves it
 * (a fresh chat opens), and opening it brings its workspace back — or, when that
 * workspace was deleted since, says so once instead. One deleted while in use (another
 * tab, the CLI) is noticed by the first request that 404s, and the default takes over.
 */
export function useWorkspaces({
  enabled,
  apiStatus,
  conversationWorkspace,
  activeConversationId,
  newConversation,
  relistCorpus,
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
  // Again once a rejected key works: the list it 401'd on was never read.
  const authorized = apiStatus !== 'unauthorized';
  useEffect(() => {
    if (enabled && authorized) void loadWorkspaces();
  }, [enabled, authorized, loadWorkspaces]);

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
    void relistCorpus();
  });

  const chooseWorkspace = useCallback(
    (name: string) => {
      if (conversationWorkspace !== null && conversationWorkspace !== name) newConversation();
      switchWorkspace(name);
    },
    [conversationWorkspace, newConversation, switchWorkspace],
  );

  const notify = useStableHandler(pushToast);

  // The workspace in use was deleted elsewhere: its requests 404. Only for a request sent
  // to the one still in use — an upload left running in a workspace since switched away
  // from doesn't move this one — and once: the requests after the switch go to the default.
  const fallBackToDefault = useStableHandler((missing: string) => {
    if (missing !== getWorkspace()) return;
    switchWorkspace('');
    void loadWorkspaces();
    notify({
      id: MISSING_TOAST_ID,
      tone: 'warn',
      title: 'Workspace no longer exists',
      body: `Workspace ${missing} was deleted; now using the default workspace.`,
    });
  });
  useEffect(() => {
    onWorkspaceMissing(fallBackToDefault);
    return () => onWorkspaceMissing(null);
  }, [fallBackToDefault]);

  // A conversation's workspace missing from the list may only be newer than the list (made
  // in another tab or with the CLI): it is read again once before calling it deleted.
  const missingNoted = useRef<string | null>(null);
  const recheckingFor = useRef<string | null>(null);
  const [recheckedFor, setRecheckedFor] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled || !loaded || conversationWorkspace === null) return;
    if (conversationWorkspace === workspace) return;
    if (conversationWorkspace === '' || workspaces.includes(conversationWorkspace)) {
      switchWorkspace(conversationWorkspace);
      return;
    }
    const key = `${activeConversationId}\n${conversationWorkspace}`;
    if (recheckedFor !== key) {
      if (recheckingFor.current === key) return;
      recheckingFor.current = key;
      // Settled either way: a list that can't be read doesn't hold the notice back.
      void loadWorkspaces().finally(() => setRecheckedFor(key));
      return;
    }
    // Deleted since. Questions here now go to the workspace in use; say so once.
    if (missingNoted.current === key) return;
    missingNoted.current = key;
    notify({
      id: MISSING_TOAST_ID,
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
    recheckedFor,
    switchWorkspace,
    loadWorkspaces,
    notify,
  ]);

  const createWorkspace = useCallback(
    async (name: string) => {
      try {
        await api.createWorkspace(name);
      } catch (err) {
        // A 401 shows the key banner, a 403 the read-only UI; the bar shows the reason.
        noteAuthFailure(err);
        throw new Error(apiErrorMessage(err, 'The workspace was not created.'));
      }
      // It exists now, whether or not the list below can be read: chosen while missing
      // from the list, the picker showed "Default" while questions went to it.
      setWorkspaces((prev) => (prev.includes(name) ? prev : [...prev, name]));
      await loadWorkspaces();
      chooseWorkspace(name);
    },
    [loadWorkspaces, chooseWorkspace, noteAuthFailure],
  );

  // Only an empty workspace can be deleted (the server says 409 otherwise).
  const deleteWorkspace = useCallback(
    async (name: string) => {
      try {
        await api.deleteWorkspace(name);
      } catch (err) {
        noteAuthFailure(err);
        throw new Error(apiErrorMessage(err, 'The workspace was not deleted.'));
      }
      setWorkspaces((prev) => prev.filter((candidate) => candidate !== name));
      chooseWorkspace('');
      await loadWorkspaces();
    },
    [loadWorkspaces, chooseWorkspace, noteAuthFailure],
  );

  return {
    workspace,
    workspaces,
    dropRememberedIfGone,
    reloadWorkspaces: loadWorkspaces,
    chooseWorkspace,
    createWorkspace,
    deleteWorkspace,
  };
}
