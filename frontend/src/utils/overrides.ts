import type { QuestionOverrides } from '../api/types';

/** No per-question retrieval override: the server's own settings apply. */
export const EMPTY_OVERRIDES: QuestionOverrides = {
  rerankTopK: null,
  maxContextChunks: null,
  llmTemperature: null,
};
