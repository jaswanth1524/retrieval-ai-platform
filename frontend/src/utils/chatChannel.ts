/** Tabs telling each other the stored history changed: localStorage's ``storage``
 *  event carries only its compact copy of the 20 most recent conversations, so a
 *  change to an older one (kept only in IndexedDB) reached another tab only on its
 *  next reload — and that tab's next save could write its stale copy back. */

const CHANNEL = 'docrag-chat';
// This tab's own announcements reach its own listener too (another channel object on
// the same page): each tab signs its messages and skips its own, which re-read and
// merged what it had just written.
const TAB = Math.random().toString(36).slice(2) + Date.now().toString(36);

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
    channel.postMessage({ type: 'saved', from: TAB });
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
    const data = event.data as { type?: unknown; from?: unknown } | null;
    if (data?.type === 'saved' && data.from !== TAB) onSaved();
  };
  return () => channel.close();
}

/** Tell the other tabs which conversation this tab is answering in now (null: none).
 *  Sent again every few seconds while it lasts, so a tab closed mid-answer stops
 *  counting once its last word is old. */
export function announceAnswering(conversationId: string | null): void {
  const channel = open();
  if (!channel) return;
  try {
    channel.postMessage({ type: 'answering', from: TAB, conversationId });
  } catch {
    // Then the other tabs don't know; nothing breaks.
  } finally {
    channel.close();
  }
}

/** ``onChange`` with another tab's id and the conversation it is answering in (null:
 *  none any more); returns the unsubscribe. */
export function onOtherTabAnswering(
  onChange: (tab: string, conversationId: string | null) => void,
): () => void {
  const channel = open();
  if (!channel) return () => undefined;
  channel.onmessage = (event: MessageEvent) => {
    const data = event.data as { type?: unknown; from?: unknown; conversationId?: unknown } | null;
    if (data?.type !== 'answering' || typeof data.from !== 'string' || data.from === TAB) return;
    onChange(data.from, typeof data.conversationId === 'string' ? data.conversationId : null);
  };
  return () => channel.close();
}
