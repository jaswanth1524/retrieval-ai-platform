import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from 'react';
import { getWorkspace } from '../api/client';
import { copyText } from '../utils/clipboard';
import { deepLinkUrl, parseDeepLink, parsePassageLink, passageLinkUrl } from '../utils/deepLink';
import type { ConversationSummary } from './chat/model';
import type { UseToastsResult } from './useToasts';

/** What the document viewer shows (App's ``sourceView``). */
export interface SourceView {
  filename: string;
  chunkId: string | null;
  highlights?: string[];
  /** ``chunkId`` is where the reader left off (Read), not a cited passage. */
  resume?: boolean;
}

interface LinksOptions {
  historyReady: boolean;
  /** The server answers with this browser's key: a passage link waits for it (opened
   *  before, its document's fetch failed with a 401, or from a workspace not yet
   *  listed). Absent: ready. */
  serverReady?: boolean;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  workspace: string;
  sourceView: SourceView | null;
  setSourceView: Dispatch<SetStateAction<SourceView | null>>;
  setPanelDrawerOpen: Dispatch<SetStateAction<boolean>>;
  chooseWorkspaceRef: { current: (name: string) => void };
  switchConversationFromList: (id: string, turnId?: string) => void;
  pushToast: UseToastsResult['push'];
}

/**
 * Links into the app: opening a conversation, answer or passage link (``#c=…``,
 * ``#d=…``) on load and on every hash change, and copying such links.
 */
export function useLinks({
  historyReady,
  serverReady = true,
  conversations,
  activeConversationId,
  workspace,
  sourceView,
  setSourceView,
  setPanelDrawerOpen,
  chooseWorkspaceRef,
  switchConversationFromList,
  pushToast,
}: LinksOptions) {
  // Links back to a conversation or an answer (#c=…&t=…): opened on load and whenever
  // the hash changes. They work in this browser only — conversations live in its
  // storage — so one from elsewhere says so instead of doing nothing.
  const conversationIdsRef = useRef<Set<string>>(new Set());
  conversationIdsRef.current = new Set(conversations.map((conversation) => conversation.id));
  useEffect(() => {
    if (!historyReady) return;
    const open = () => {
      // A document or passage on this server (#d=…&k=…): opened in the viewer.
      const passage = parsePassageLink(window.location.hash);
      // Left on the address until the server answers: opened then (this runs again).
      if (passage && !serverReady) return;
      if (passage) {
        window.history.replaceState(null, '', window.location.pathname + window.location.search);
        if ((passage.workspace ?? '') !== (getWorkspace() ?? '')) {
          chooseWorkspaceRef.current(passage.workspace ?? '');
        }
        setSourceView({ filename: passage.filename, chunkId: passage.chunkId ?? null });
        setPanelDrawerOpen(false);
        return;
      }
      const link = parseDeepLink(window.location.hash);
      if (!link) return;
      // Taken off the address once read: reloading shouldn't jump back to it.
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
      if (!conversationIdsRef.current.has(link.conversationId)) {
        pushToast({
          tone: 'warn',
          title: 'That conversation isn’t in this browser',
          body: 'Links open conversations saved here; it may have been deleted.',
        });
        return;
      }
      switchConversationFromList(link.conversationId, link.turnId);
    };
    open();
    window.addEventListener('hashchange', open);
    return () => window.removeEventListener('hashchange', open);
    // The setters and the ref never change identity: listed only to satisfy the rule.
  }, [historyReady, serverReady, pushToast, switchConversationFromList, setSourceView, setPanelDrawerOpen, chooseWorkspaceRef]);
  const copyPassageLink = useCallback(() => {
    if (!sourceView) return;
    const url = passageLinkUrl(
      {
        filename: sourceView.filename,
        ...(sourceView.chunkId ? { chunkId: sourceView.chunkId } : {}),
        ...(workspace ? { workspace } : {}),
      },
      window.location,
    );
    void copyText(url).then((ok) =>
      pushToast(
        ok
          ? { tone: 'good', title: 'Link copied', body: 'It opens this passage for anyone with access to this server.' }
          : { tone: 'warn', title: 'Could not copy the link', body: url },
      ),
    );
  }, [sourceView, workspace, pushToast]);
  const copyLink = useCallback(
    (turnId?: string) => {
      if (!activeConversationId) return;
      const url = deepLinkUrl({ conversationId: activeConversationId, turnId }, window.location);
      void copyText(url).then((ok) =>
        pushToast(
          ok
            ? { tone: 'good', title: 'Link copied', body: 'It opens this conversation in this browser.' }
            : { tone: 'warn', title: 'Could not copy the link', body: url },
        ),
      );
    },
    [activeConversationId, pushToast],
  );

  return { copyPassageLink, copyLink };
}
