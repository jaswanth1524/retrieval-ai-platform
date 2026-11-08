import type { CitationResponse } from '../api/types';
import type { AnswerVersion, ChatTurn } from '../components/ChatMessage';

/** Earlier answers kept on a regenerated one, at most. */
export const MAX_ANSWER_VERSIONS = 5;

/** An answer as an earlier version: its text, its sources (without their passage text,
 *  which the viewer fetches again) and when and by which model it was written. */
export function asVersion(turn: ChatTurn): AnswerVersion {
  return {
    content: turn.content,
    sources: turn.sources.map(withoutText),
    timestamp: turn.timestamp,
    ...(turn.model ? { model: turn.model } : {}),
  };
}

function withoutText(source: CitationResponse): CitationResponse {
  const { spans: _spans, ...rest } = source;
  return { ...rest, text: '' };
}

/** ``turns`` once a regenerated answer (``newAnswerId``) is done: when the answer it
 *  replaces (``oldAnswerId``) and its question sit right before the new question and
 *  answer, the old pair goes and the old answer is kept as an earlier version of the
 *  new one. Anything else (another question in between, the old one gone) is left as
 *  it is — the new pair simply follows. */
export function mergeRegenerated(
  turns: ChatTurn[],
  oldAnswerId: string,
  newAnswerId: string,
): ChatTurn[] {
  const index = turns.findIndex((turn) => turn.id === oldAnswerId);
  if (index < 1) return turns;
  const [oldQuestion, oldAnswer, newQuestion, newAnswer] = turns.slice(index - 1, index + 3);
  if (
    oldQuestion?.role !== 'user' ||
    oldAnswer?.role !== 'assistant' ||
    newQuestion?.role !== 'user' ||
    newAnswer?.id !== newAnswerId ||
    oldQuestion.content !== newQuestion.content
  ) {
    return turns;
  }
  const versions = [...(oldAnswer.versions ?? []), asVersion(oldAnswer)].slice(-MAX_ANSWER_VERSIONS);
  return [
    ...turns.slice(0, index - 1),
    newQuestion,
    { ...newAnswer, versions },
    ...turns.slice(index + 3),
  ];
}
