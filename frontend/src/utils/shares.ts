import type { ChatTurn } from '../components/ChatMessage';
import type { CitationResponse, SharedConversationResponse, SharedTurn } from '../api/types';

const SHARE_ID = /^[A-Za-z0-9_-]{8,64}$/;

/** ``#s=<id>``: a shared answer or conversation on this server. */
export function parseShareLink(hash: string): string | null {
  const id = new URLSearchParams(hash.replace(/^#/, '')).get('s');
  return id && SHARE_ID.test(id) ? id : null;
}

export function shareLinkUrl(id: string, location: Pick<Location, 'origin' | 'pathname'>): string {
  return `${location.origin}${location.pathname}#s=${encodeURIComponent(id)}`;
}

function answerStatus(turn: ChatTurn): string {
  if (turn.notFound) return 'insufficient_context';
  if (turn.uncited) return 'uncited';
  return 'answered';
}

function sharedSource(source: CitationResponse): CitationResponse {
  return {
    source_number: source.source_number,
    filename: source.filename,
    page: source.page,
    ...(source.page_end != null ? { page_end: source.page_end } : {}),
    section: source.section,
    chunk_id: source.chunk_id,
    text: source.text,
    ...(source.spans?.length ? { spans: source.spans } : {}),
    ...(source.rerank_score != null ? { rerank_score: source.rerank_score } : {}),
  };
}

/** The question/answer pairs of ``turns`` as shared: each answer with the question
 *  asked before it, its sources and how it was answered — no notes, ratings, usage or
 *  trace ids. ``onlyTurnId``: just that answer. Stopped or failed answers are left out. */
export function sharedTurns(turns: ChatTurn[], onlyTurnId?: string): SharedTurn[] {
  const shared: SharedTurn[] = [];
  let question = '';
  for (const turn of turns) {
    if (turn.role === 'user') {
      question = turn.content;
      continue;
    }
    if (turn.role !== 'assistant' || !turn.content.trim() || turn.stopped || turn.incomplete) {
      continue;
    }
    if (onlyTurnId && turn.id !== onlyTurnId) continue;
    shared.push({
      question,
      answer: turn.content,
      sources: turn.sources.map(sharedSource),
      answer_status: answerStatus(turn),
      ...(turn.model ? { model: turn.model } : {}),
      asked_at: turn.timestamp / 1000,
      truncated: Boolean(turn.truncated),
    });
  }
  return shared.slice(-100);
}

/** Passages a share's sources may fill in at most (one request each). */
export const SHARE_FILL_MAX = 40;

/** ``shared`` with the passage text older turns had dropped (a browser keeps it only for
 *  its latest answers) read back from the server: each source whose ``text`` is empty
 *  is looked up by its chunk id. One still not found (re-indexed, deleted) stays empty;
 *  a failed lookup never stops the share. */
export async function withPassageText(
  shared: SharedTurn[],
  readChunk: (filename: string, chunkId: string) => Promise<string | null>,
): Promise<SharedTurn[]> {
  const wanted = new Map<string, Promise<string | null>>();
  for (const turn of shared) {
    for (const source of turn.sources) {
      if (source.text || !source.chunk_id) continue;
      const key = `${source.filename}\u0000${source.chunk_id}`;
      if (!wanted.has(key) && wanted.size < SHARE_FILL_MAX) {
        wanted.set(
          key,
          readChunk(source.filename, source.chunk_id).catch(() => null),
        );
      }
    }
  }
  if (wanted.size === 0) return shared;
  const found = new Map<string, string | null>();
  await Promise.all(
    [...wanted].map(async ([key, text]) => {
      found.set(key, await text);
    }),
  );
  return shared.map((turn) => ({
    ...turn,
    sources: turn.sources.map((source) => {
      if (source.text) return source;
      const text = found.get(`${source.filename}\u0000${source.chunk_id}`);
      return text ? { ...source, text } : source;
    }),
  }));
}

/** A share as a conversation of this browser ("Import as conversation"): each shared
 *  answer after its question, flagged as it was answered. Ids are given on import. */
export function turnsFromShare(share: SharedConversationResponse): ChatTurn[] {
  const turns: ChatTurn[] = [];
  share.conversation.forEach((shared, index) => {
    const timestamp = Math.round((shared.asked_at ?? share.created_at) * 1000);
    if (shared.question) {
      turns.push({
        id: `q${index}`,
        role: 'user',
        content: shared.question,
        sources: [],
        timestamp,
        timings: null,
        traceId: null,
      });
    }
    const status = shared.answer_status ?? 'answered';
    turns.push({
      id: `a${index}`,
      role: 'assistant',
      content: shared.answer,
      sources: shared.sources,
      timestamp,
      timings: null,
      traceId: null,
      ...(shared.model ? { model: shared.model } : {}),
      ...(status === 'insufficient_context' || status === 'no_context' ? { notFound: true } : {}),
      ...(status === 'uncited' ? { uncited: true } : {}),
      ...(shared.truncated ? { truncated: true } : {}),
    });
  });
  return turns;
}
