import type { HistoryMessage } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { sliceWhole } from './text';

// Mirrors the server's own REQUEST_HISTORY_MAX_MESSAGES cap (api/schemas.py) — kept
// in sync so a client-side trim never silently disagrees with what the server would
// accept anyway. The char budget is a client-only practical guard against crowding
// a small local model's context window; the server has its own count-based backstop.
const HISTORY_MAX_MESSAGES = 12;
const HISTORY_CHAR_BUDGET = 8000;
// Mirrors the server's REQUEST_HISTORY_MESSAGE_MAX_CHARS (api/schemas.py) per-message
// cap. Without this, one assistant answer over 4000 chars survives the total-budget
// trim below (it's the only message so far) and the server 422s the whole request.
const HISTORY_MESSAGE_MAX_CHARS = 4000;
// How far back from the cap a whitespace break is still worth taking. A hard cut lands
// mid-word and can sever a trailing qualifier or negation, which reads to the model as
// something the speaker never said; backing up to the last space avoids that. Bounded
// because text with no whitespace in its tail (a base64 blob, unspaced CJK) would
// otherwise back off arbitrarily far — past this, a clean cut beats a short message.
// A summary or comparison answer sent back as history: its gist is what a follow-up
// ("what does it say about deadlines?") needs, and at the full 4000 characters one
// summary took half of HISTORY_CHAR_BUDGET from the exchanges before it.
const SUMMARY_HISTORY_MAX_CHARS = 1500;

/** Truncate to HISTORY_MESSAGE_MAX_CHARS, preferring a nearby whitespace boundary.
 *
 * The result is always <= HISTORY_MESSAGE_MAX_CHARS: the ellipsis has to fit inside the
 * server's per-message limit too, so the budget it occupies is reserved up front rather
 * than appended after a full-width cut (which would land one char over and 422). */
function truncateForHistory(content: string, maxChars = HISTORY_MESSAGE_MAX_CHARS): string {
  if (content.length <= maxChars) return content;
  const ELLIPSIS = '…';
  const budget = maxChars - ELLIPSIS.length;
  const hardCut = sliceWhole(content, budget);
  const lastBreak = hardCut.lastIndexOf(' ');
  const backoff = Math.floor(maxChars * 0.12);
  const cut = lastBreak >= budget - backoff ? hardCut.slice(0, lastBreak) : hardCut;
  // The marker is for the model's benefit: without it a truncated answer looks complete,
  // so it may treat a severed sentence as the speaker's final word.
  return `${cut.trimEnd()}${ELLIPSIS}`;
}

function isCompleteAnswer(turn: ChatTurn | undefined): boolean {
  return (
    turn?.role === 'assistant' &&
    turn.content.trim().length > 0 &&
    // A stopped answer is a fragment; sent back as history it reads to the model as
    // what it actually concluded. Same for one cut short by a failed connection.
    !turn.stopped &&
    !turn.incomplete &&
    // "Not enough information" answered nothing: sent back, the server's rewrite of the
    // next question folded the refusal into the search and the model refused again.
    !turn.notFound
  );
}

/** Build bounded conversation history from prior turns: only answered questions and
 * their answers (a question that failed, was stopped or is being asked again has no
 * answer, and sent alone it read to the model as two questions in a row), most recent
 * first up to HISTORY_MAX_MESSAGES, each truncated to HISTORY_MESSAGE_MAX_CHARS, then
 * trimmed to a total char budget (dropping the oldest first), restored to
 * chronological order. */
export function buildHistory(turns: ChatTurn[]): HistoryMessage[] {
  const candidates = turns
    .filter((turn, index): turn is ChatTurn & { role: 'user' | 'assistant' } =>
      turn.role === 'user'
        ? turn.content.trim().length > 0 && isCompleteAnswer(turns[index + 1])
        : isCompleteAnswer(turn) && turns[index - 1]?.role === 'user',
    )
    .slice(-HISTORY_MAX_MESSAGES);

  const withinBudget: HistoryMessage[] = [];
  let charsUsed = 0;
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    const turn = candidates[i];
    const isSummary = turn.role === 'assistant' && Boolean(candidates[i - 1]?.task);
    const content = truncateForHistory(turn.content, isSummary ? SUMMARY_HISTORY_MAX_CHARS : undefined);
    charsUsed += content.length;
    if (charsUsed > HISTORY_CHAR_BUDGET && withinBudget.length > 0) break;
    withinBudget.unshift({ role: candidates[i].role, content });
  }
  // The count cap can split a pair: history never opens on an answer to a question
  // it doesn't carry.
  if (withinBudget[0]?.role === 'assistant') withinBudget.shift();
  return withinBudget;
}

/** History for asking ``question`` again (Retry, Regenerate): the exchange it repeats is
 *  left out, or the model saw the question twice, the second time already answered. */
export function historyForQuestion(turns: ChatTurn[], question: string): HistoryMessage[] {
  const history = buildHistory(turns);
  const last = history.length - 2;
  if (last >= 0 && history[last].role === 'user' && history[last].content === question) {
    return history.slice(0, last);
  }
  return history;
}
