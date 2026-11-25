/** Tabs telling each other the stored history changed: localStorage's ``storage``
 *  event carries only its compact copy of the 20 most recent conversations, so a
 *  change to an older one (kept only in IndexedDB) reached another tab only on its
 *  next reload — and that tab's next save could write its stale copy back. */

const CHANNEL = 'docrag-chat';

function open(): BroadcastChannel | null {
  try {
    return typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(CHANNEL);
  } catch {
    return null;
  }
}

/** Tell the other tabs the full history was just saved. */
export function announceHistorySaved(): void {
  const channel = open();
  if (!channel) return;
  try {
    channel.postMessage({ type: 'saved' });
  } catch {
    // Nothing to tell then; their next load reads it.
  } finally {
    channel.close();
  }
}

/** ``onSaved`` whenever another tab saved the full history; returns the unsubscribe. */
export function onHistorySaved(onSaved: () => void): () => void {
  const channel = open();
  if (!channel) return () => undefined;
  channel.onmessage = (event: MessageEvent) => {
    if ((event.data as { type?: unknown } | null)?.type === 'saved') onSaved();
  };
  return () => channel.close();
}
