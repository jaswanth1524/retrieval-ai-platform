import { useCallback, useState } from 'react';
import type {
  DocumentTask,
  LlmProvider,
  PassageNarrowing,
  QuestionOverrides,
} from '../api/types';
import { scopeForAskAbout } from '../appHelpers';
import { QUESTION_INPUT_ID } from '../utils/id';
import { MAX_SCOPE_FILENAMES } from '../utils/limits';
import type { ConversationScope } from './chatPersistence';
import type { UseChatResult } from './useChat';
import { useStableHandler } from './useStableHandler';
import type { ToastInput } from './useToasts';

export interface AskSettings {
  provider: LlmProvider;
  overrides: QuestionOverrides;
  /** The conversation's documents (undefined = the whole corpus) and tags. */
  scopeFilenames: string[] | undefined;
  selectedTags: string[];
  /** The scope's page range and upload dates, as a narrowing. */
  scopePages: PassageNarrowing | undefined;
  modelLabel: string | undefined;
}

export interface AskActionsDeps {
  ask: UseChatResult['ask'];
  editAndResend: UseChatResult['editAndResend'];
  setScope: (scope: ConversationScope) => void;
  pushToast: (toast: ToastInput) => string;
  /** Narrow screens: the context panel is a drawer, closed (or opened) around an ask. */
  panelCollapsed: boolean;
  setPanelDrawerOpen: (open: boolean) => void;
  closeSourceView: () => void;
  /** The inspector's pin (a stable state setter): an edit answers in a new conversation. */
  setPinnedTraceId: (traceId: string | null) => void;
  /** Asking (summaries, picked passages) waits: booting, refused key, an answer streaming. */
  summarizeDisabled: boolean;
}

/** Every way a question is asked in the open conversation: the composer, Retry and
 *  Regenerate, "ask across all documents", picked passages, an edit, a summary. */
export function useAskActions(settings: AskSettings, deps: AskActionsDeps) {
  const { provider, overrides, scopeFilenames, selectedTags, scopePages, modelLabel } = settings;
  const { ask, editAndResend, setScope, pushToast, panelCollapsed, setPanelDrawerOpen } = deps;

  // Shared by the input box and a Retry click on a failed turn — retry always uses the
  // CURRENT provider/overrides/scope, not whatever was selected when the original
  // question failed. useCallback (not a plain const) because this is passed down as
  // ChatMessage's onRetry prop through ChatThread — an unstable reference here would
  // defeat React.memo(ChatMessage) and reintroduce a full-list re-render on every SSE
  // delta. `task`: a summary or comparison (its own documents; the scope doesn't
  // apply). `bypassCache`: Regenerate must produce a new answer, not the cached copy of
  // the one on screen.
  const askWith = useCallback(
    (
      question: string,
      task: DocumentTask | undefined,
      narrowing: PassageNarrowing | undefined,
      bypassCache: boolean,
    ) => {
      const looking = narrowing ?? scopePages;
      // Picked passages are the scope: the conversation's documents and tags are AND-ed
      // with them on the server, so a passage from another document (the viewer's
      // "Ask about this") found nothing.
      const picked = Boolean(looking?.chunk_ids?.length) && !task;
      return ask(question, provider, overrides, picked ? undefined : scopeFilenames, {
        tags: picked ? [] : selectedTags,
        ...(bypassCache ? { bypassCache: true } : {}),
        modelLabel,
        ...(task ? { task } : {}),
        ...(looking && !task ? { narrowing: looking } : {}),
      });
    },
    [ask, provider, overrides, scopeFilenames, selectedTags, modelLabel, scopePages],
  );
  const askQuestion = useCallback(
    (question: string, task?: DocumentTask, narrowing?: PassageNarrowing) =>
      askWith(question, task, narrowing, false),
    [askWith],
  );
  const regenerateQuestion = useCallback(
    (question: string, task?: DocumentTask, narrowing?: PassageNarrowing) =>
      askWith(question, task, narrowing, true),
    [askWith],
  );
  // A "not found" answer asked within some documents: the same question over all of
  // them, with the conversation's scope cleared so the next question searches them too.
  const askEverywhere = useCallback(
    (question: string) => {
      setScope({ filenames: [], tags: [] });
      return ask(question, provider, overrides, undefined, { tags: [], modelLabel });
    },
    [ask, setScope, provider, overrides, modelLabel],
  );
  // From the Search panel: the search's question over only the passages ticked there,
  // in the open conversation (the passages are its scope; they were found within it).
  const askAboutPassages = useStableHandler((question: string, chunkIds: string[]) => {
    void askQuestion(question, undefined, { chunk_ids: chunkIds });
    if (panelCollapsed) setPanelDrawerOpen(false);
  });
  // A question about one passage, from the document viewer: asked over it alone.
  const askAboutViewedPassage = useStableHandler((question: string, chunkId: string) => {
    deps.closeSourceView();
    askAboutPassages(question, [chunkId]);
  });
  // One document summarized, or two compared, as a turn in the open conversation.
  // Stable: every memoized document card receives it, and it used to change with the
  // scope and the provider (askQuestion's), re-rendering all of them.
  // Opens the focus dialog first ("deadlines", "what changed"); it runs on Enter.
  const [summaryFilenames, setSummaryFilenames] = useState<string[] | null>(null);
  const summarizeDocuments = useStableHandler((filenames: string[]) => {
    if (deps.summarizeDisabled || filenames.length === 0) return;
    setSummaryFilenames([...new Set(filenames)]);
  });
  const runSummary = useStableHandler((filenames: string[], instruction?: string) => {
    setSummaryFilenames(null);
    if (deps.summarizeDisabled) return;
    const mode: DocumentTask['mode'] =
      filenames.length === 1 ? 'summarize' : filenames.length === 2 ? 'compare' : 'brief';
    const task: DocumentTask = { mode, filenames, ...(instruction ? { instruction } : {}) };
    const label =
      mode === 'compare'
        ? `Compare ${filenames[0]} and ${filenames[1]}`
        : mode === 'brief'
          ? `Brief on ${filenames.length} documents: ${filenames.join(', ')}`
          : `Summarize ${filenames[0]}`;
    const shown = instruction ? `${label} — focus: ${instruction}` : label;
    // On a narrow screen the corpus panel is a drawer over the conversation.
    setPanelDrawerOpen(false);
    void askQuestion(shown, task);
  });
  // "Ask about these": the open conversation's scope becomes those documents, and the
  // question box takes focus. Before, the composer's scope popover was the only way.
  const askAbout = useStableHandler((filenames: string[]) => {
    const { filenames: scoped, selected } = scopeForAskAbout(filenames);
    setScope({ filenames: scoped, tags: [] });
    setPanelDrawerOpen(false);
    pushToast(
      scoped.length < selected
        ? {
            tone: 'warn',
            title: `Asking about the first ${scoped.length} of ${selected} documents`,
            body: `A question can be limited to at most ${MAX_SCOPE_FILENAMES} documents; ask without a selection to search them all.`,
          }
        : {
            tone: 'good',
            title:
              scoped.length === 1 ? `Asking about ${scoped[0]}` : `Asking about ${scoped.length} documents`,
          },
    );
    window.requestAnimationFrame(() => document.getElementById(QUESTION_INPUT_ID)?.focus());
  });
  const { setPinnedTraceId } = deps;
  const editQuestion = useCallback(
    (turnId: string, question: string) => {
      setPinnedTraceId(null);
      return editAndResend(turnId, question, provider, overrides, scopeFilenames, {
        tags: selectedTags,
        modelLabel,
        ...(scopePages ? { narrowing: scopePages } : {}),
      });
    },
    [editAndResend, setPinnedTraceId, provider, overrides, scopeFilenames, selectedTags, modelLabel, scopePages],
  );

  return {
    askQuestion,
    regenerateQuestion,
    askEverywhere,
    askAboutPassages,
    askAboutViewedPassage,
    summaryFilenames,
    setSummaryFilenames,
    summarizeDocuments,
    runSummary,
    askAbout,
    editQuestion,
  };
}
