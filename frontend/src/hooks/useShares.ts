import { useCallback, useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import type { SharedTurn } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { copyText } from '../utils/clipboard';
import { parseShareLink, sharedTurns, shareLinkUrl, withPassageText } from '../utils/shares';
import type { UseToastsResult } from './useToasts';

interface SharesOptions {
  /** SHARES_ENABLED: off, nothing is offered (a #s= link still says why it can't open). */
  enabled: boolean;
  turns: ChatTurn[];
  conversationTitle: string;
  pushToast: UseToastsResult['push'];
}

/** A cited passage's text as indexed now (null: no longer there). */
async function readChunk(filename: string, chunkId: string): Promise<string | null> {
  const content = await api.getDocumentContent(filename, undefined, { around: chunkId, radius: 0 });
  return content.chunks.find((chunk) => chunk.chunk_id === chunkId)?.text ?? null;
}

/** Shared answers: sharing one (or the conversation) by link, and opening a share link
 *  (``#s=<id>``) on load and on every hash change. */
export function useShares({ enabled, turns, conversationTitle, pushToast }: SharesOptions) {
  const [openShareId, setOpenShareId] = useState<string | null>(null);
  useEffect(() => {
    const open = () => {
      const id = parseShareLink(window.location.hash);
      if (!id) return;
      // Taken off the address once read, as conversation links are.
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
      setOpenShareId(id);
    };
    open();
    window.addEventListener('hashchange', open);
    return () => window.removeEventListener('hashchange', open);
  }, []);

  const publish = useCallback(
    async (shared: SharedTurn[], title: string) => {
      if (shared.length === 0) {
        pushToast({ tone: 'warn', title: 'Nothing to share yet', body: 'Ask a question first.' });
        return;
      }
      try {
        const turns = await withPassageText(shared, readChunk);
        const share = await api.createShare({ title: title.slice(0, 200) || 'Shared answer', turns });
        const link = shareLinkUrl(share.id, window.location);
        const copied = await copyText(link);
        pushToast({
          tone: 'good',
          title: copied ? 'Share link copied' : 'Shared',
          body: copied
            ? 'Anyone whose key may use this workspace can open it.'
            : `Copy this link: ${link}`,
        });
      } catch (err) {
        pushToast({ tone: 'warn', title: 'Could not share', body: apiErrorMessage(err, 'Try again.') });
      }
    },
    [pushToast],
  );

  const shareAnswer = useCallback(
    (turnId: string) => {
      const shared = sharedTurns(turns, turnId);
      void publish(shared, shared[0]?.question || conversationTitle);
    },
    [turns, conversationTitle, publish],
  );

  const shareConversation = useCallback(() => {
    void publish(sharedTurns(turns), conversationTitle);
  }, [turns, conversationTitle, publish]);

  return {
    openShareId,
    closeShare: useCallback(() => setOpenShareId(null), []),
    shareAnswer: enabled ? shareAnswer : undefined,
    shareConversation: enabled ? shareConversation : undefined,
  };
}
