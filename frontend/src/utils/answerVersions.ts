import type { CitationResponse } from '../api/types';
import type { AnswerVersion, ChatTurn } from '../components/ChatMessage';

/** Earlier answers kept on a regenerated one, at most. */
export const MAX_ANSWER_VERSIONS = 5;
/** Ids of replaced answers an answer remembers (links and pinned traces to them). */
const MAX_PREVIOUS_IDS = 20;

/** An answer as an earlier version: its text, its sources (without their passage text,
 *  which the viewer fetches again), when and by which model it was written, and the
 *  style it was asked for. */
export function asVersion(turn: ChatTurn): AnswerVersion {
  return {
    content: turn.content,
    sources: turn.sources.map(withoutText),
    timestamp: turn.timestamp,
    ...(turn.model ? { model: turn.model } : {}),
    ...(turn.style ? { style: turn.style } : {}),
    ...(turn.traceId ? { traceId: turn.traceId } : {}),
  };
}

function withoutText(source: CitationResponse): CitationResponse {
  const { spans: _spans, ...rest } = source;
  return { ...rest, text: '' };
}

/** ``turns`` once a regenerated answer (``newAnswerId``) is done. Its question and
 *  answer were added at the end; the answer it replaces (``oldAnswerId``) and that
 *  answer's question go, and the new pair takes their place — right there when the old
 *  pair was the last one before it, and at the old pair's position when it was an
 *  earlier answer (Regenerate on any answer replaces that answer, not the latest one).
 *  The old answer is kept as an earlier version; it stays saved if it was, and its id
 *  is remembered (``previousIds``), so a link or a pinned trace naming it still finds
 *  the answer. Anything else (another question asked, the old one gone) is left as it
 *  is — the new pair simply follows. */
export function mergeRegenerated(
  turns: ChatTurn[],
  oldAnswerId: string,
  newAnswerId: string,
): ChatTurn[] {
  const index = turns.findIndex((turn) => turn.id === oldAnswerId);
  const end = turns.length;
  const oldQuestion = turns[index - 1];
  const oldAnswer = turns[index];
  const newQuestion = turns[end - 2];
  const newAnswer = turns[end - 1];
  if (
    index < 1 ||
    index > end - 3 ||
    oldQuestion?.role !== 'user' ||
    oldAnswer?.role !== 'assistant' ||
    newQuestion?.role !== 'user' ||
    newAnswer?.id !== newAnswerId ||
    oldQuestion.content !== newQuestion.content
  ) {
    return turns;
  }
  const versions = [...(oldAnswer.versions ?? []), asVersion(oldAnswer)].slice(-MAX_ANSWER_VERSIONS);
  const previousIds = [...(oldAnswer.previousIds ?? []), oldAnswer.id].slice(-MAX_PREVIOUS_IDS);
  const merged: ChatTurn = {
    ...newAnswer,
    versions,
    previousIds,
    ...(oldAnswer.saved ? { saved: true } : {}),
  };
  return [...turns.slice(0, index - 1), newQuestion, merged, ...turns.slice(index + 1, end - 2)];
}

/** The answer an id names now: itself, or the answer that replaced it. */
export function findAnswer(turns: ChatTurn[], id: string): ChatTurn | undefined {
  return turns.find((turn) => turn.id === id) ?? turns.find((turn) => turn.previousIds?.includes(id));
}

const FORMAT_LABELS: Record<string, string> = { bullets: 'bullets', table: 'table', brief: 'shorter' };

/** How a version was asked for ("table", "German"), or "" for the plain answer. */
export function styleLabel(version: Pick<AnswerVersion, 'style'>): string {
  const parts = [
    version.style?.answerFormat ? FORMAT_LABELS[version.style.answerFormat] : undefined,
    version.style?.answerLanguage,
  ].filter(Boolean);
  return parts.join(', ');
}


// What belongs to one answer's run, not to its text: dropped when another version
// becomes the answer (they described the run that wrote the answer being replaced).
const RUN_FIELDS = [
  'timings',
  'cached',
  'notFound',
  'uncited',
  'weakMatch',
  'truncated',
  'autoCited',
  'topRerankScore',
  'rerankMinScore',
  'usage',
  'followups',
  'contextSources',
  'passagesLeftOut',
  'irrelevantSources',
  'coverage',
  'reviewedAt',
  // A rating was of the other answer's text.
  'feedback',
] as const;

/** ``turn`` with its earlier version ``index`` as the answer again; the answer it
 *  had becomes that version (so nothing is lost, and the order of the others stays). */
export function withVersionRestored(turn: ChatTurn, index: number): ChatTurn {
  const versions = turn.versions ?? [];
  const chosen = versions[index];
  if (!chosen) return turn;
  const rest: Partial<ChatTurn> = { ...turn };
  for (const field of RUN_FIELDS) delete rest[field];
  delete rest.model;
  delete rest.style;
  const restored: ChatTurn = {
    ...(rest as ChatTurn),
    content: chosen.content,
    sources: chosen.sources,
    timestamp: chosen.timestamp,
    timings: null,
    traceId: chosen.traceId ?? null,
    ...(chosen.model ? { model: chosen.model } : {}),
    ...(chosen.style ? { style: chosen.style } : {}),
    versions: versions.map((version, at) => (at === index ? asVersion(turn) : version)),
  };
  return restored;
}
