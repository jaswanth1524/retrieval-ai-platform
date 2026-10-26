// App-level constants and pure helpers (split out of App.tsx).
import type { CitationResponse, PublicConfigResponse, QuestionOverrides } from './api/types';
import type { ChatMode } from './components/ChatHeader';
import type { ChatTurn } from './components/ChatMessage';
import type { DocumentMeta } from './components/CorpusPanel';
import type { RailPanel } from './components/IconRail';
import { MAX_SCOPE_FILENAMES } from './utils/limits';
import { EMPTY_OVERRIDES, isAnswerFormat, isAnswerLanguage } from './utils/overrides';
import { readStored, readStoredJson } from './utils/safeStorage';

export const ADVANCED_OPTIONS_STORAGE_KEY = 'docrag-advanced-options';
export const MODE_STORAGE_KEY = 'docrag-mode';
export const PROVIDER_STORAGE_KEY = 'docrag-provider';
// Also read by main.tsx before the first render.
export const THEME_STORAGE_KEY = 'docrag-theme';

export function loadPersistedOverrides(): QuestionOverrides {
  // Storage denied/corrupt — fall back to defaults for this session.
  const parsed = readStoredJson<Partial<QuestionOverrides>>(ADVANCED_OPTIONS_STORAGE_KEY);
  if (!parsed) return EMPTY_OVERRIDES;
  return {
    rerankTopK: parsed.rerankTopK ?? null,
    maxContextChunks: parsed.maxContextChunks ?? null,
    llmTemperature: parsed.llmTemperature ?? null,
    answerFormat: isAnswerFormat(parsed.answerFormat) ? parsed.answerFormat : null,
    answerLanguage:
      typeof parsed.answerLanguage === 'string' && isAnswerLanguage(parsed.answerLanguage)
        ? parsed.answerLanguage.trim()
        : null,
  };
}

/** Drop saved overrides the server would now reject.
 *
 * Overrides are saved in the browser, limits live on the server: an operator lowering
 * FUSED_TOP_N below a saved "Rerank top K" made every question fail with a 422 — and
 * the slider, clamped by the browser to the new max, couldn't show why. */
export function clampOverrides(
  overrides: QuestionOverrides,
  config: PublicConfigResponse,
): QuestionOverrides {
  const within = (value: number | null, max: number, min = 1) =>
    value !== null && (value < min || value > max) ? null : value;
  const rerankTopK = within(overrides.rerankTopK, config.rerank_top_k_limit);
  const maxContextChunks = within(overrides.maxContextChunks, config.max_context_chunks_limit);
  return {
    rerankTopK,
    // More context chunks than reranked ones is a 422 on every question (the server's
    // QuestionRequest rule), and the values were saved, so it outlived a reload.
    maxContextChunks:
      rerankTopK !== null && maxContextChunks !== null
        ? Math.min(maxContextChunks, rerankTopK)
        : maxContextChunks,
    llmTemperature: within(overrides.llmTemperature, config.llm_temperature_max, 0),
    answerFormat: overrides.answerFormat ?? null,
    answerLanguage: overrides.answerLanguage ?? null,
  };
}

export function loadPersistedMode(): ChatMode {
  return readStored(MODE_STORAGE_KEY) === 'engineer' ? 'engineer' : 'reader';
}

// The browser's own PDF viewer shows an original in the document viewer; past this a
// download serves better than holding the whole file in memory as a blob.
const MAX_PREVIEW_BYTES = 50 * 1024 * 1024;

// How long a pointer or focus rests on a citation before its viewer page is fetched.
export const CITATION_PREFETCH_DWELL_MS = 150;
// An answer whose best passage scored under RERANK_MIN_SCORE plus this is flagged
// "weakly supported": it cleared the bar, but only just.
export const WEAK_SUPPORT_MARGIN = 0.1;
// A failed document listing is retried from this delay, doubling up to BOOT_RETRY_MAX_MS.
export const LISTING_RETRY_BASE_MS = 5000;
// Backoff between boot attempts while the API is unreachable: 2s, 4s, 8s ... 30s.
export const BOOT_RETRY_BASE_MS = 2000;
export const BOOT_RETRY_MAX_MS = 30_000;

export function canPreviewOriginal(
  filename: string,
  originalsStored: boolean,
  meta: DocumentMeta | undefined,
): boolean {
  return (
    originalsStored &&
    filename.toLowerCase().endsWith('.pdf') &&
    Boolean(meta?.reindexable) &&
    (meta?.bytes ?? 0) <= MAX_PREVIEW_BYTES
  );
}

// Shared empty values: a fresh [] per render would re-render the memoized child.
/** The rail panels to leave out: feedback and the query log are operator data (full
 *  key), and each exists only when the server has it on. */
export function hiddenPanels(feedbackAvailable: boolean, unansweredAvailable: boolean): RailPanel[] {
  return [...(feedbackAvailable ? [] : ['feedback' as const]), ...(unansweredAvailable ? [] : ['unanswered' as const])];
}
export const NO_SOURCES: CitationResponse[] = [];

export const PANEL_META: Record<RailPanel, { title: string; actionLabel: string }> = {
  chat: { title: 'Conversations', actionLabel: 'New' },
  corpus: { title: 'Corpus', actionLabel: 'Upload' },
  search: { title: 'Search passages', actionLabel: 'Clear' },
  traces: { title: 'Traces', actionLabel: 'Refresh' },
  feedback: { title: 'Feedback', actionLabel: 'Refresh' },
  unanswered: { title: 'Asked questions', actionLabel: 'Refresh' },
};

export function scopeLabel(
  indexedFilenames: string[],
  selectedFilenames: string[],
  selectedTags: string[] = [],
): string {
  const tagPart = selectedTags.length > 0 ? ` · #${selectedTags.join(' #')}` : '';
  if (selectedFilenames.length === 0) {
    return selectedTags.length > 0
      ? `Tagged${tagPart}`
      : `All ${indexedFilenames.length} documents`;
  }
  if (selectedFilenames.length === 1) return `${selectedFilenames[0]}${tagPart}`;
  return `${selectedFilenames.length} documents${tagPart}`;
}

/** Why a finished "not found" answer has no sources, for the inspector's Sources tab
 *  (which otherwise said to ask a question, under an answer that had been asked). */
export function notFoundSourcesMessage(turn: ChatTurn | undefined): string | undefined {
  if (!turn?.notFound) return undefined;
  const read = turn.contextSources?.length ?? 0;
  if (read > 0) {
    return `The answer cited nothing: it read ${read} passage${read === 1 ? '' : 's'} and none answered the question. They are listed under the answer.`;
  }
  return turn.scoped
    ? 'Nothing in the selected documents matched this question.'
    : 'Nothing in your documents matched this question.';
}

/** The documents "Ask about these" limits a conversation to: each once, at most the
 *  server's MAX_SCOPE_FILENAMES (past it every question and search was a 422), and how
 *  many were selected in all. */
export function scopeForAskAbout(filenames: readonly string[]): {
  filenames: string[];
  selected: number;
} {
  const unique = [...new Set(filenames)];
  return { filenames: unique.slice(0, MAX_SCOPE_FILENAMES), selected: unique.length };
}

/** What a conversation file holds: "Export all conversations" wrote several (each with
 *  its title), a single conversation's export is one; ``none`` when it isn't JSON. */
export function parseConversationFile(
  text: string,
):
  | { kind: 'many'; items: { raw: unknown; title?: string }[] }
  | { kind: 'one'; raw: unknown }
  | { kind: 'none' } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { kind: 'none' };
  }
  const many =
    parsed && typeof parsed === 'object' && 'docrag_conversations' in parsed
      ? (parsed as { docrag_conversations: unknown }).docrag_conversations
      : null;
  if (Array.isArray(many)) {
    return {
      kind: 'many',
      items: many.map((item: unknown) => ({
        raw: item,
        title:
          item && typeof item === 'object' && typeof (item as { title?: unknown }).title === 'string'
            ? (item as { title: string }).title
            : undefined,
      })),
    };
  }
  return parsed === null ? { kind: 'none' } : { kind: 'one', raw: parsed };
}
