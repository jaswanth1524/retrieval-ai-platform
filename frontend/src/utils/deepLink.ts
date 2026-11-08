/** Links to a conversation or one answer in it: ``#c=<conversation>&t=<turn>``. They
 *  open in this browser only — conversations live in its storage — so a link is for
 *  coming back (a bookmark, a note), not for sending to someone else. */

const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

export interface DeepLink {
  conversationId: string;
  turnId?: string;
}

export function parseDeepLink(hash: string): DeepLink | null {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const conversationId = params.get('c');
  if (!conversationId || !ID_PATTERN.test(conversationId)) return null;
  const turnId = params.get('t');
  return turnId && ID_PATTERN.test(turnId) ? { conversationId, turnId } : { conversationId };
}

export function deepLinkHash(link: DeepLink): string {
  const params = new URLSearchParams({ c: link.conversationId });
  if (link.turnId) params.set('t', link.turnId);
  return `#${params.toString()}`;
}

export function deepLinkUrl(link: DeepLink, location: Pick<Location, 'origin' | 'pathname'>): string {
  return `${location.origin}${location.pathname}${deepLinkHash(link)}`;
}

/** Tokens and estimated cost of a conversation's answers (null when none reported). */
export function usageTotal(
  turns: ReadonlyArray<{ usage?: { prompt: number; completion: number; costUsd?: number } }>,
): { tokens: number; costUsd?: number } | null {
  let tokens = 0;
  let cost = 0;
  let priced = false;
  let any = false;
  for (const turn of turns) {
    if (!turn.usage) continue;
    any = true;
    tokens += turn.usage.prompt + turn.usage.completion;
    if (turn.usage.costUsd !== undefined) {
      priced = true;
      cost += turn.usage.costUsd;
    }
  }
  if (!any) return null;
  return priced ? { tokens, costUsd: cost } : { tokens };
}
