/** Reading history back: everything rendered is validated, anything damaged dropped. */
import type {
  CitationResponse,
  DocumentTask,
  PassageNarrowing,
  SummaryCoverage,
} from '../../api/types';
import type { AnswerStyle, ChatTurn } from '../../components/ChatMessage';
import { isAnswerFormat, isAnswerLanguage } from '../../utils/overrides';
import { WORKSPACE_NAME_PATTERN } from '../../utils/limits';
import {
  cleanLabels,
  type Conversation,
  type ConversationScope,
  type DeletedConversation,
  MAX_TIMESTAMP_SECONDS,
  deriveTitle,
} from './model';

export function isChatTurn(value: unknown): value is ChatTurn {
  if (!value || typeof value !== 'object') return false;
  const turn = value as Partial<ChatTurn>;
  return (
    typeof turn.id === 'string' &&
    (turn.role === 'user' || turn.role === 'assistant' || turn.role === 'error') &&
    typeof turn.content === 'string' &&
    Array.isArray(turn.sources) &&
    typeof turn.timestamp === 'number'
  );
}
// The range a Date can hold. A number outside it (an import with `"timestamp": 1e300`)
// passed the type check, and then every render threw "Invalid time value" on it.
const MAX_TIME_MS = 8.64e15;
/** ``value`` when it is a time a Date can format, else ``fallback``. */
export function validTime(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= MAX_TIME_MS
    ? value
    : fallback;
}
function isCitation(value: unknown): value is CitationResponse {
  if (!value || typeof value !== 'object') return false;
  const source = value as Partial<CitationResponse>;
  return (
    typeof source.source_number === 'number' &&
    typeof source.filename === 'string' &&
    typeof source.page === 'number' &&
    typeof source.section === 'string' &&
    typeof source.chunk_id === 'string' &&
    typeof source.text === 'string'
  );
}
/** The citation, without its spans when any of them isn't one: the citation still opens
 *  its source, just with nothing marked. */
function withValidSpans(source: CitationResponse): CitationResponse {
  // A page range is shown as text: anything but a whole number is dropped.
  if (source.page_end != null && !Number.isInteger(source.page_end)) {
    const { page_end: _pageEnd, ...rest } = source;
    return withValidSpans(rest);
  }
  if (source.spans === undefined) return source;
  const valid =
    Array.isArray(source.spans) &&
    source.spans.every(
      (span: unknown) =>
        !!span &&
        typeof span === 'object' &&
        typeof (span as { start?: unknown }).start === 'number' &&
        typeof (span as { end?: unknown }).end === 'number' &&
        typeof (span as { quote?: unknown }).quote === 'string',
    );
  if (valid) return source;
  const { spans: _spans, ...rest } = source;
  return rest;
}
/** A turn read from disk or an imported file, with every field the UI renders checked.
 *
 * Only the top-level shape used to be checked, so an imported file with `model: {}` or
 * a `null` source was saved — and crashed the page on every load once it rendered. */
// The server sends at most 5 follow-ups of at most 200 characters (api/prompting.py).
const MAX_FOLLOWUPS = 5;
const FOLLOWUP_MAX_CHARS = 200;
/** Suggested follow-ups as the thread can render them: strings with text, each once —
 *  they are the buttons' React keys, so a repeated one (a model can suggest the same
 *  question twice) broke rendering. From the stream and from disk alike. */
export function cleanFollowups(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const followups = (value as unknown[]).filter(
    (item): item is string =>
      typeof item === 'string' && item.trim().length > 0 && item.length <= FOLLOWUP_MAX_CHARS,
  );
  return [...new Set(followups)].slice(0, MAX_FOLLOWUPS);
}
/** ``value`` as a document task when it is one the server would accept, else null. */
export function sanitizeTask(value: unknown): DocumentTask | null {
  if (!value || typeof value !== 'object') return null;
  const task = value as Record<string, unknown>;
  const mode = task.mode;
  if (mode !== 'summarize' && mode !== 'compare' && mode !== 'brief') return null;
  const names = task.filenames;
  if (!Array.isArray(names) || !names.every((name) => typeof name === 'string' && name !== '')) {
    return null;
  }
  const countOk =
    mode === 'brief' ? names.length >= 2 && names.length <= 50 : names.length === (mode === 'compare' ? 2 : 1);
  if (!countOk || new Set(names).size !== names.length) {
    return null;
  }
  const sanitized: DocumentTask = { mode, filenames: [...(names as string[])] };
  if (typeof task.instruction === 'string' && task.instruction.trim()) {
    sanitized.instruction = task.instruction;
  }
  // A page range only narrows a summary of one document.
  if (mode === 'summarize') {
    if (isPage(task.pageMin)) sanitized.pageMin = task.pageMin;
    if (isPage(task.pageMax)) sanitized.pageMax = task.pageMax;
  }
  return sanitized;
}
function isPage(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 1_000_000;
}

/** A question's saved narrowing, within what the server takes (schemas.PassageNarrowing),
 *  or null when nothing usable is left. */
export function sanitizeNarrowing(value: unknown): PassageNarrowing | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const page = (item: unknown) =>
    typeof item === 'number' && Number.isInteger(item) && item >= 1 ? item : undefined;
  const strings = (list: unknown, max: number, maxChars: number) =>
    Array.isArray(list)
      ? list
          .filter((item): item is string => typeof item === 'string' && item !== '' && item.length <= maxChars)
          .slice(0, max)
      : [];
  const narrowing: PassageNarrowing = {};
  const pageMin = page(raw.page_min);
  const pageMax = page(raw.page_max);
  if (pageMin !== undefined && !(pageMax !== undefined && pageMin > pageMax)) narrowing.page_min = pageMin;
  if (pageMax !== undefined && !(pageMin !== undefined && pageMin > pageMax)) narrowing.page_max = pageMax;
  if (typeof raw.section === 'string' && raw.section && raw.section.length <= 200) {
    narrowing.section = raw.section;
  }
  const exclude = strings(raw.exclude_filenames, 500, 1024);
  if (exclude.length > 0) narrowing.exclude_filenames = exclude;
  const chunkIds = strings(raw.chunk_ids, 100, 64);
  if (chunkIds.length > 0) narrowing.chunk_ids = chunkIds;
  const left = strings(raw.exclude_chunk_ids, 100, 64);
  if (left.length > 0) narrowing.exclude_chunk_ids = left;
  const time = (item: unknown) =>
    typeof item === 'number' && Number.isFinite(item) && item >= 0 && item <= MAX_TIMESTAMP_SECONDS
      ? item
      : undefined;
  const after = time(raw.uploaded_after);
  const before = time(raw.uploaded_before);
  if (after !== undefined && !(before !== undefined && after > before)) narrowing.uploaded_after = after;
  if (before !== undefined && !(after !== undefined && after > before)) narrowing.uploaded_before = before;
  return Object.keys(narrowing).length > 0 ? narrowing : null;
}
/** A summary's coverage is kept on its turn only when it says something: a sampled
 *  read, a part left out, notes cut off. */
export function coverageWorthShowing(coverage: SummaryCoverage): boolean {
  return (
    coverage.read < coverage.total || (coverage.skipped ?? 0) > 0 || (coverage.cutOff ?? 0) > 0
  );
}
function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}
export function sanitizeTurn(value: unknown): ChatTurn | null {
  if (!isChatTurn(value)) return null;
  const turn = value as ChatTurn & Record<string, unknown>;
  const timings = turn.timings as unknown;
  const sanitized: ChatTurn = {
    id: turn.id,
    role: turn.role,
    content: turn.content,
    timestamp: validTime(turn.timestamp, Date.now()),
    sources: turn.sources.filter(isCitation).map(withValidSpans),
    timings:
      timings && typeof timings === 'object' && typeof (timings as { total_ms?: unknown }).total_ms === 'number'
        ? (timings as ChatTurn['timings'])
        : null,
    traceId: typeof turn.traceId === 'string' ? turn.traceId : null,
  };
  if (typeof turn.question === 'string') sanitized.question = turn.question;
  if (turn.feedback === 'up' || turn.feedback === 'down') sanitized.feedback = turn.feedback;
  if (typeof turn.model === 'string') sanitized.model = turn.model;
  const flags = [
    'stopped',
    'incomplete',
    'cached',
    'notFound',
    'uncited',
    'truncated',
    'followUp',
    'weakMatch',
    'autoCited',
    'scoped',
  ] as const;
  for (const flag of flags) {
    if (turn[flag] === true) sanitized[flag] = true;
  }
  if (Array.isArray(turn.contextSources)) {
    const context = turn.contextSources.filter(isCitation).map(withValidSpans).slice(0, 20);
    if (context.length > 0) sanitized.contextSources = context;
  }
  if (Array.isArray(turn.versions) && turn.role === 'assistant') {
    const versions = turn.versions.flatMap((version: unknown) => {
      if (!version || typeof version !== 'object') return [];
      const { content, sources, timestamp, model, style, traceId } = version as Record<
        string,
        unknown
      >;
      if (typeof content !== 'string' || !Array.isArray(sources)) return [];
      const versionStyle = sanitizeStyle(style);
      return [
        {
          content: content.slice(0, 200_000),
          sources: sources.filter(isCitation).slice(0, 50),
          timestamp: validTime(timestamp, sanitized.timestamp),
          ...(typeof model === 'string' && model ? { model: model.slice(0, 200) } : {}),
          ...(versionStyle ? { style: versionStyle } : {}),
          ...(typeof traceId === 'string' && traceId && traceId.length <= 100 ? { traceId } : {}),
        },
      ];
    });
    if (versions.length > 0) sanitized.versions = versions.slice(-5);
  }
  const task = sanitizeTask(turn.task);
  if (task) sanitized.task = task;
  const narrowing = sanitizeNarrowing(turn.narrowing);
  if (narrowing) sanitized.narrowing = narrowing;
  const followups = cleanFollowups(turn.followups);
  if (followups.length > 0) sanitized.followups = followups;
  if (turn.saved === true && turn.role === 'assistant') sanitized.saved = true;
  if (turn.role === 'assistant' && typeof turn.reviewedAt === 'number' && turn.reviewedAt > 0) {
    sanitized.reviewedAt = validTime(turn.reviewedAt, sanitized.timestamp);
  }
  const style = turn.role === 'assistant' ? sanitizeStyle(turn.style) : undefined;
  if (style) sanitized.style = style;
  if (Array.isArray(turn.previousIds) && turn.role === 'assistant') {
    const ids = turn.previousIds.filter(
      (id: unknown): id is string => typeof id === 'string' && id.length > 0 && id.length <= 100,
    );
    if (ids.length > 0) sanitized.previousIds = ids.slice(-20);
  }
  if (
    typeof turn.passagesLeftOut === 'number' &&
    Number.isInteger(turn.passagesLeftOut) &&
    turn.passagesLeftOut > 0 &&
    turn.passagesLeftOut <= 1000
  ) {
    sanitized.passagesLeftOut = turn.passagesLeftOut;
  }
  if (Array.isArray(turn.irrelevantSources)) {
    const numbers = turn.irrelevantSources.filter(
      (value): value is number => Number.isInteger(value) && value >= 1 && value <= 1000,
    );
    if (numbers.length > 0) sanitized.irrelevantSources = [...new Set(numbers)].slice(0, 50);
  }
  if (typeof turn.topRerankScore === 'number' && turn.topRerankScore >= 0 && turn.topRerankScore <= 1) {
    sanitized.topRerankScore = turn.topRerankScore;
  }
  if (typeof turn.rerankMinScore === 'number' && turn.rerankMinScore >= 0 && turn.rerankMinScore <= 1) {
    sanitized.rerankMinScore = turn.rerankMinScore;
  }
  const usage = turn.usage as unknown;
  if (usage && typeof usage === 'object') {
    const { prompt, completion, costUsd } = usage as Record<string, unknown>;
    if (isCount(prompt) && isCount(completion)) {
      sanitized.usage = { prompt, completion };
      if (typeof costUsd === 'number' && Number.isFinite(costUsd) && costUsd >= 0) {
        sanitized.usage.costUsd = costUsd;
      }
    }
  }
  const coverage = turn.coverage as unknown;
  if (coverage && typeof coverage === 'object') {
    const { read, total, skipped, cutOff } = coverage as Record<string, unknown>;
    if (isCount(read) && isCount(total)) {
      const kept: SummaryCoverage = {
        read,
        total,
        ...(isCount(skipped) && skipped > 0 ? { skipped } : {}),
        ...(isCount(cutOff) && cutOff > 0 ? { cutOff } : {}),
      };
      if (coverageWorthShowing(kept)) sanitized.coverage = kept;
    }
  }
  return sanitized;
}
/** The recorded deletes in a saved v2 payload (either store). */
export function parseDeleted(value: unknown): DeletedConversation[] | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const deleted = (value as { deleted?: unknown }).deleted;
  if (!Array.isArray(deleted)) return undefined;
  return deleted.filter(
    (entry): entry is DeletedConversation =>
      !!entry &&
      typeof entry === 'object' &&
      typeof (entry as DeletedConversation).id === 'string' &&
      typeof (entry as DeletedConversation).at === 'number',
  );
}
/** The conversations in a saved v2 payload (either store), or null when unreadable. */
export function parseConversations(value: unknown): Conversation[] | null {
  if (!value || typeof value !== 'object') return null;
  const parsed = value as { version?: number; conversations?: unknown };
  if (parsed.version !== 2 || !Array.isArray(parsed.conversations)) return null;
  return parsed.conversations
    .map(sanitizeConversation)
    .filter((conversation): conversation is Conversation => conversation !== null);
}
export function sanitizeConversation(value: unknown): Conversation | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Partial<Conversation>;
  if (typeof raw.id !== 'string' || !Array.isArray(raw.turns)) return null;
  const turns = raw.turns.map(sanitizeTurn).filter((turn): turn is ChatTurn => turn !== null);
  const conversation: Conversation = {
    id: raw.id,
    title: typeof raw.title === 'string' && raw.title ? raw.title : deriveTitle(turns),
    createdAt: validTime(raw.createdAt, Date.now()),
    updatedAt: validTime(raw.updatedAt, Date.now()),
    turns,
    scope: sanitizeScope(raw.scope),
  };
  if (validTime(raw.editedAt, NaN) === raw.editedAt) conversation.editedAt = raw.editedAt;
  if (raw.pinned === true) conversation.pinned = true;
  const labels = cleanLabels(raw.labels);
  if (labels.length > 0) conversation.labels = labels;
  if (typeof raw.workspace === 'string' && WORKSPACE_NAME_PATTERN.test(raw.workspace)) {
    conversation.workspace = raw.workspace;
  }
  return conversation;
}
export function sanitizeScope(value: unknown): ConversationScope | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = value as Partial<ConversationScope>;
  const strings = (list: unknown) =>
    Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string') : [];
  const page = (item: unknown) =>
    typeof item === 'number' && Number.isInteger(item) && item >= 1 && item <= 1_000_000 ? item : undefined;
  const scope: ConversationScope = { filenames: strings(raw.filenames), tags: strings(raw.tags) };
  const pageMin = page(raw.pageMin);
  const pageMax = page(raw.pageMax);
  if (pageMin !== undefined && (pageMax === undefined || pageMin <= pageMax)) scope.pageMin = pageMin;
  if (pageMax !== undefined && (pageMin === undefined || pageMin <= pageMax)) scope.pageMax = pageMax;
  const stamp = (item: unknown) =>
    typeof item === 'number' && Number.isFinite(item) && item >= 0 && item <= MAX_TIMESTAMP_SECONDS
      ? item
      : undefined;
  const after = stamp(raw.uploadedAfter);
  const before = stamp(raw.uploadedBefore);
  if (after !== undefined) scope.uploadedAfter = after;
  if (before !== undefined) scope.uploadedBefore = before;
  return scope;
}

/** An answer's Reformat style, within what Settings → Answers offers; undefined for
 *  none (or nothing usable). */
export function sanitizeStyle(value: unknown): AnswerStyle | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const { answerFormat, answerLanguage } = value as Record<string, unknown>;
  const style: AnswerStyle = {
    ...(isAnswerFormat(answerFormat) && answerFormat !== 'default' ? { answerFormat } : {}),
    ...(typeof answerLanguage === 'string' && isAnswerLanguage(answerLanguage) && answerLanguage
      ? { answerLanguage }
      : {}),
  };
  return Object.keys(style).length > 0 ? style : undefined;
}
