import type { AnswerFormat, QuestionOverrides } from '../api/types';

/** No per-question retrieval override: the server's own settings apply. */
export const EMPTY_OVERRIDES: QuestionOverrides = {
  rerankTopK: null,
  maxContextChunks: null,
  llmTemperature: null,
  answerFormat: null,
  answerLanguage: null,
};

export const ANSWER_FORMATS: AnswerFormat[] = ['default', 'bullets', 'table', 'brief'];

export function isAnswerFormat(value: unknown): value is AnswerFormat {
  return typeof value === 'string' && (ANSWER_FORMATS as string[]).includes(value);
}

/** A language the server takes (schemas.normalize_answer_language): "match", or a
 *  name of letters and single spaces, at most 40 characters. */
export function isAnswerLanguage(value: string): boolean {
  const trimmed = value.trim();
  return (
    trimmed.length > 0 &&
    trimmed.length <= 40 &&
    (trimmed.toLowerCase() === 'match' || /^\p{L}+(?: \p{L}+)*$/u.test(trimmed))
  );
}
