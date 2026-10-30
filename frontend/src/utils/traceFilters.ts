/** Which answers the list shows, by answer status (filtered on the server, so an
 *  archive of thousands of traces returns its matching ones, not the newest only). */
export type AnswerFilter = 'any' | 'answered' | 'uncited' | 'unanswered';

export const ANSWER_FILTER_STATUSES: Record<AnswerFilter, readonly string[]> = {
  any: [],
  answered: ['answered'],
  uncited: ['uncited'],
  unanswered: ['insufficient_context', 'no_context'],
};

