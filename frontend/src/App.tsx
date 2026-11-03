import { lazy, startTransition, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, apiErrorMessage } from './api/client';
import type { LlmProvider, PublicConfigResponse, QuestionOverrides } from './api/types';
import ChatHeader from './components/ChatHeader';
import type { ChatMode } from './components/ChatHeader';
import ChatThread from './components/ChatThread';
import type { FeedbackPayload } from './components/ChatMessage';
import Composer from './components/Composer';
import { NEW_CONVERSATION_DRAFT, pruneDrafts } from './utils/drafts';
import { QUESTION_INPUT_ID } from './utils/id';
import ContextPanel from './components/ContextPanel';
import ConversationList from './components/ConversationList';
import CorpusPanel from './components/CorpusPanel';
import WorkspaceBar from './components/corpus/WorkspaceBar';
import IconRail from './components/IconRail';
import LazyChunkBoundary from './components/LazyChunkBoundary';
import { loadMarkdown } from './components/markdownLoader';
import type { RailPanel } from './components/IconRail';
import type { InspectorTab } from './components/Inspector';
import SourcePreview from './components/SourcePreview';
import ToastRow from './components/ToastRow';
import SearchPanel, { type SearchRequestFromElsewhere } from './components/SearchPanel';
import TracesPanel from './components/TracesPanel';
import SavedPanel from './components/SavedPanel';
import FeedbackPanel from './components/FeedbackPanel';
import UnansweredPanel from './components/UnansweredPanel';
import { MAX_CONVERSATIONS, MAX_PINNED_CONVERSATIONS } from './hooks/chatPersistence';
import { useChat } from './hooks/useChat';
import { useScopePresets } from './hooks/useScopePresets';
import { useStableHandler } from './hooks/useStableHandler';
import { useCommands } from './hooks/useCommands';
import { useServerHealth } from './hooks/useServerHealth';
import ServerNotices from './components/ServerNotices';
import SummaryDialog from './components/SummaryDialog';
import ShortcutsDialog from './components/ShortcutsDialog';
import ServerActivity from './components/corpus/ServerActivity';
import {
  EMPTY_CORPUS_POLL_MS,
  WATCHED_CORPUS_POLL_MS,
  useCorpusAutoRefresh,
} from './hooks/useCorpusAutoRefresh';
import { useAskActions } from './hooks/useAskActions';
import { useCitationPreview } from './hooks/useCitationPreview';
import { useConversationExports } from './hooks/useConversationExports';
import { useConversationScope } from './hooks/useConversationScope';
import { useInspectorTarget } from './hooks/useInspectorTarget';
import { useRailLists } from './hooks/useRailLists';
import { useThemePreference } from './hooks/useThemePreference';
import { useAppBoot } from './hooks/useAppBoot';
import { useWorkspaces } from './hooks/useWorkspaces';
import { useBackupActions } from './hooks/useBackupActions';
import { useInspectorVisibility } from './hooks/useInspectorVisibility';
import { useCorpus } from './hooks/useCorpus';
import { useResponsiveLayout } from './hooks/useResponsiveLayout';
import { useToasts } from './hooks/useToasts';
import { useWindowFileDrop } from './hooks/useWindowFileDrop';
import { NOTIFY_STORAGE_KEY, useBackgroundNotice } from './hooks/useBackgroundNotice';
import { sourceChange } from './utils/citationFreshness';
import { copyText } from './utils/clipboard';
import { withCitationFootnotes } from './utils/exportChat';
import { folderPickNote, pickFolderFiles } from './utils/folderFiles';
import { initialProvider } from './utils/providers';
import { readStored, writeStored } from './utils/safeStorage';
import {
  acceptedExtensions,
  DEFAULT_MAX_UPLOAD_BYTES,
  validateUploads,
} from './utils/uploadValidation';
import { sliceWhole } from './utils/text';
import {
  ADVANCED_OPTIONS_STORAGE_KEY,
  MODE_STORAGE_KEY,
  PROVIDER_STORAGE_KEY,
  loadPersistedOverrides,
  clampOverrides,
  loadPersistedMode,
  WEAK_SUPPORT_MARGIN,
  canPreviewOriginal,
  hiddenPanels,
  NO_SOURCES,
  PANEL_META,
  notFoundSourcesMessage,
  parseConversationFile,
} from './appHelpers';

// Dialogs and the inspector render only on demand, so they load on demand too: the
// first paint (thread + composer) no longer waits on their code.
const CommandPalette = lazy(() => import('./components/CommandPalette'));
const DocumentViewer = lazy(() => import('./components/DocumentViewer'));
const Inspector = lazy(() => import('./components/Inspector'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));

function App() {
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>('ollama');
  const [advancedOptions, setAdvancedOptions] = useState<QuestionOverrides>(loadPersistedOverrides);
  const { theme, toggleTheme } = useThemePreference();
  const [mode, setModeState] = useState<ChatMode>(loadPersistedMode);
  const [rail, setRail] = useState<RailPanel>('chat');
  const [searchOpened, setSearchOpened] = useState(false);
  if (rail === 'search' && !searchOpened) setSearchOpened(true);
  const [searchResetKey, setSearchResetKey] = useState(0);
  // A search started from elsewhere (a "not found" answer's "Search the passages").
  const [searchRequest, setSearchRequest] = useState<SearchRequestFromElsewhere | null>(null);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('sources');
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const {
    toasts,
    push: pushToast,
    dismiss: dismissToast,
    hold: holdToast,
    release: releaseToast,
  } = useToasts();
  const [notifyOnAnswer, setNotifyOnAnswer] = useState(
    () => readStored(NOTIFY_STORAGE_KEY) === 'true',
  );
  const restoreInputRef = useRef<HTMLInputElement>(null);
  const { tooNarrowForInspector, roomyEnoughForInspector, panelCollapsed } = useResponsiveLayout();
  // Reopening on a new turn should follow the conversation again rather than resurface
  // whatever trace row was last clicked.
  const { inspectorOpen, showInspector, closeInspector, toggleInspector } = useInspectorVisibility(
    tooNarrowForInspector,
    roomyEnoughForInspector,
    () => setPinnedTraceId(null),
  );
  // Narrow screens only: whether the context panel's drawer is showing. On wider screens
  // the panel is a grid column and this is ignored.
  const [panelDrawerOpen, setPanelDrawerOpen] = useState(false);
  // Show a rail panel, opening the drawer it is in on a narrow screen: setting the rail
  // alone left dropped files' progress, rejections and Retry in a closed drawer.
  const showPanel = useStableHandler((panel: RailPanel) => {
    setRail(panel);
    if (panelCollapsed) setPanelDrawerOpen(true);
  });
  const { hoveredCitation, setHoveredCitation, handleCitationHover, handleCitationLeave } =
    useCitationPreview();
  // Source document viewer: which document/chunk a clicked citation opens (null = closed).
  // chunkId null: opened from the corpus panel's Read, at the document's first passage.
  // highlights: the passages a citation rests on, marked in the cited chunk.
  const [sourceView, setSourceView] = useState<{
    filename: string;
    chunkId: string | null;
    highlights?: string[];
  } | null>(null);
  const corpusBrowseInputRef = useRef<HTMLInputElement>(null);
  const {
    turns,
    pending,
    streamingConversationId,
    askedTurns,
    stage,
    stageTiming,
    ask,
    cancel,
    clear: clearTurns,
    conversations,
    savedAnswers,
    answersCiting,
    activeConversationId,
    newConversation: createConversation,
    switchConversation: selectConversation,
    renameConversation,
    deleteConversation: removeConversation,
    setPinned,
    searchConversations,
    persistError,
    persistPartial,
    setTurnFeedback,
    setSourceIrrelevant,
    setTurnSaved,
    scope,
    conversationWorkspace,
    setScope,
    pruneScopes,
    removeFromScopes,
    editAndResend,
    importConversation,
    importConversations,
    importPreview,
    historyReady,
    allConversationsJson,
  } = useChat();
  // A deleted conversation's unsent draft goes with it, whichever tab deleted it. After
  // the composer has saved it on the switch away, hence an effect on the list. Not
  // before the full history has loaded: a conversation only IndexedDB holds would
  // look deleted, and lose its draft.
  useEffect(() => {
    if (!historyReady) return;
    pruneDrafts(new Set(conversations.map((conversation) => conversation.id)));
  }, [conversations, historyReady]);
  // At the cap, starting another conversation drops the least recently used unpinned
  // one. Said once a session, when the history first fills, rather than not at all.
  const capNoticeShownRef = useRef(false);
  useEffect(() => {
    if (!historyReady || capNoticeShownRef.current) return;
    if (conversations.length < MAX_CONVERSATIONS) return;
    capNoticeShownRef.current = true;
    pushToast({
      tone: 'warn',
      title: `History is full (${MAX_CONVERSATIONS} conversations)`,
      body: 'Each new conversation now replaces the least recently used one that isn’t pinned. Pin or export the ones to keep.',
    });
  }, [conversations.length, historyReady, pushToast]);
  // An answer can stream into one conversation while another is open. The open one shows
  // its progress (and offers Stop) only when it is the one being answered.
  const streamingHere = pending && streamingConversationId === activeConversationId;
  // Which turn the inspector is describing (see useInspectorTarget).
  const { setPinnedTraceId, inspectedTurn, inspectedTraceId, inspectedTraceReady } =
    useInspectorTarget(turns, streamingHere);
  const {
    uploads,
    indexedFilenames,
    documents,
    availableTags,
    chunkTotal,
    refreshDocuments,
    handleUpload,
    uploadTags,
    setUploadTags,
    handleRetryUpload,
    handleDismissUpload,
    handleCancelUpload,
    handleReindex,
    handleRefreshFromUrl,
    handleSetRefreshInterval,
    handleRestoreVersion,
    handleUploadFromUrl,
    handleReindexMany,
    handleReindexAllStale,
    handleDeleteDocument,
    handleSetTags,
    trackJobs,
    resetForWorkspace,
  } = useCorpus({
    onAuthFailure: (err) => noteAuthFailure(err),
    onAuthRestored: () => noteAuthRestored(),
    pushToast,
    pruneScopes,
    removeFromScopes,
  });

  // The server's config, key and reachability; the corpus's first listing.
  const applyConfig = (configResult: PublicConfigResponse) => {
    setAdvancedOptions((prev) => {
      const clamped = clampOverrides(prev, configResult);
      const changed =
        clamped.rerankTopK !== prev.rerankTopK ||
        clamped.maxContextChunks !== prev.maxContextChunks ||
        clamped.llmTemperature !== prev.llmTemperature;
      if (!changed) return prev;
      writeStored(ADVANCED_OPTIONS_STORAGE_KEY, JSON.stringify(clamped));
      return clamped;
    });
    // A provider picked earlier in this browser wins while it can answer; otherwise
    // the server's default, then any other that can (never stuck on an Ollama that
    // is down while OpenAI is configured). The remembered choice stays stored.
    const startOn = initialProvider(readStored(PROVIDER_STORAGE_KEY), configResult);
    if (startOn) setSelectedProvider(startOn);
  };
  const {
    apiStatus,
    corpusChecked,
    listingFailed,
    config,
    access,
    noteAuthFailure,
    noteAuthRestored,
    refreshAccess,
    relistCorpus,
    retryBoot,
  } = useAppBoot({
    refreshDocuments,
    onConfig: applyConfig,
    beforeListing: (signal) => dropRememberedIfGone(signal),
  });
  const readOnly = access === 'read';
  const workspacesEnabled = Boolean(config?.workspaces_enabled);
  const {
    workspace,
    workspaces,
    dropRememberedIfGone,
    reloadWorkspaces,
    chooseWorkspace,
    createWorkspace,
    deleteWorkspace,
  } = useWorkspaces({
    enabled: workspacesEnabled,
    apiStatus,
    conversationWorkspace,
    activeConversationId,
    newConversation: () => newConversation(),
    relistCorpus,
    noteAuthFailure,
    pushToast,
    // Nothing of the previous workspace stays on screen to act on: its corpus (and the
    // panel's selection, keyed below), a viewer of its file, its search results.
    onSwitched: () => {
      resetForWorkspace();
      // A trace pinned in the inspector belongs to the previous workspace.
      setPinnedTraceId(null);
      setSourceView(null);
      setSearchResetKey((key) => key + 1);
    },
  });
  // The corpus panel's changes go to the workspace it was showing when they started: a
  // bulk delete or tag loop keeps these for its whole run (the panel is keyed by
  // workspace), so a switch halfway doesn't send the rest to the next workspace.
  const here = useMemo(
    () => ({
      deleteDocument: (filename: string) => handleDeleteDocument(filename, workspace),
      setTags: (filename: string, tags: string[]) => handleSetTags(filename, tags, workspace),
      refresh: (filename: string) => void handleRefreshFromUrl(filename, workspace),
      setRefreshInterval: (filename: string, minutes: number | null) =>
        void handleSetRefreshInterval(filename, minutes, workspace),
      restoreVersion: (filename: string, version: string) =>
        void handleRestoreVersion(filename, version, workspace),
      reindex: (filename: string) => void handleReindex(filename, workspace),
      reindexMany: (filenames: string[]) => void handleReindexMany(filenames, workspace),
    }),
    [
      handleDeleteDocument,
      handleSetTags,
      handleRefreshFromUrl,
      handleSetRefreshInterval,
      handleRestoreVersion,
      handleReindex,
      handleReindexMany,
      workspace,
    ],
  );
  // Each upload card is shown in its own workspace: one still indexing (or failed) in
  // another keeps going there, and is back on the way back.
  const workspaceUploads = useMemo(
    () => uploads.filter((item) => (item.workspace ?? '') === workspace),
    [uploads, workspace],
  );
  // This tab's own uploads in flight: the activity list shows only the rest.
  const followedKey = workspaceUploads
    .filter((item) => item.status === 'uploading')
    .map((item) => item.filename)
    .join('\n');
  const followedUploads = useMemo(
    () => new Set(followedKey ? followedKey.split('\n') : []),
    [followedKey],
  );
  const relistAfterOtherJobs = useStableHandler(() => {
    void refreshDocuments().catch(noteAuthFailure);
  });
  const { exportingBackup, restoring, downloadBackup, restoreBackup, downloadOriginal } =
    useBackupActions({
      pushToast,
      dismissToast,
      noteAuthFailure,
      onRestoreQueued: async (jobs, restoredTo) => {
        showPanel('corpus');
        await trackJobs(jobs, restoredTo);
      },
    });
  // The search scope belongs to the active conversation (see useChat).
  const {
    selectedFilenames,
    selectedTags,
    setSelectedFilenames,
    setSelectedTags,
    pageRange,
    setPageRange,
    setUploadedAfter,
    setUploadedBefore,
    scopePages,
    scopeFilenames,
    currentScopeLabel,
  } = useConversationScope(scope, setScope, indexedFilenames);

  // A pinned trace belongs to one conversation's turns, so anything that replaces the
  // turns array has to release it. Otherwise inspectedTurn's lookup finds nothing, the
  // Retrieval/Trace tabs fall back to the stale pinnedTraceId and keep rendering the
  // previous conversation's trace, while the Sources tab (which reads the live turn)
  // correctly goes empty — one inspector showing two different questions. The hovered
  // citation goes too: its button unmounts without a mouseleave/blur (a switch from the
  // keyboard or palette), which left its preview on screen over the next conversation.
  // Stable (useChat's actions are), so the memoized conversation list skips re-rendering
  // on every streamed token.
  const clear = useCallback(() => {
    setPinnedTraceId(null);
    setHoveredCitation(null);
    clearTurns();
  }, [clearTurns, setPinnedTraceId, setHoveredCitation]);
  const newConversation = useCallback(() => {
    setPinnedTraceId(null);
    setHoveredCitation(null);
    createConversation();
  }, [createConversation, setPinnedTraceId, setHoveredCitation]);
  const switchConversation = useCallback(
    (id: string) => {
      setPinnedTraceId(null);
      setHoveredCitation(null);
      // A transition: a long conversation mounts up to 100 rendered answers, and the
      // list's highlight and the drawer closing shouldn't wait for all of them.
      startTransition(() => selectConversation(id));
    },
    [selectConversation, setPinnedTraceId, setHoveredCitation],
  );
  const deleteConversation = useCallback(
    (id: string) => {
      setPinnedTraceId(null);
      setHoveredCitation(null);
      removeConversation(id);
    },
    [removeConversation, setPinnedTraceId, setHoveredCitation],
  );
  // A conversation search hit names the turn it matched: the thread scrolls to it.
  const [focusTurn, setFocusTurn] = useState<{ turnId: string } | null>(null);
  const switchConversationFromList = useCallback(
    (id: string, turnId?: string) => {
      switchConversation(id);
      setFocusTurn(turnId ? { turnId } : null);
      setPanelDrawerOpen(false);
    },
    [switchConversation],
  );
  const pinConversation = useCallback(
    (id: string, pinned: boolean) => {
      if (!setPinned(id, pinned)) {
        pushToast({
          tone: 'warn',
          title: 'Too many pinned conversations',
          body: `Unpin one first — at most ${MAX_PINNED_CONVERSATIONS} stay pinned.`,
        });
      }
    },
    [setPinned, pushToast],
  );

  const updateAdvancedOptions = (next: QuestionOverrides) => {
    setAdvancedOptions(next);
    writeStored(ADVANCED_OPTIONS_STORAGE_KEY, JSON.stringify(next));
  };

  const setMode = useCallback((next: ChatMode) => {
    setModeState(next);
    writeStored(MODE_STORAGE_KEY, next);
  }, []);

  // The markdown renderer is its own chunk; fetch it once the first paint is done, so
  // it is there before the first answer is.
  useEffect(() => {
    const preload = () => void loadMarkdown();
    if (typeof window.requestIdleCallback === 'function') {
      const handle = window.requestIdleCallback(preload, { timeout: 2000 });
      return () => window.cancelIdleCallback(handle);
    }
    const timer = setTimeout(preload, 200);
    return () => clearTimeout(timer);
  }, []);

  const {
    feedbackItems,
    feedbackError,
    loadFeedback,
    unanswered,
    unansweredError,
    loadUnanswered,
    traces,
    tracesError,
    loadTraces,
    traceAnswerFilter,
    setTraceAnswerFilter,
  } = useRailLists(rail, workspace, noteAuthFailure);

  // The feedback panel needs feedback on and the full key (GET /feedback is operator data).
  const feedbackPanelAvailable = Boolean(config?.feedback_enabled) && !readOnly;
  useEffect(() => {
    if (rail === 'feedback' && !feedbackPanelAvailable) setRail('chat');
  }, [rail, feedbackPanelAvailable]);

  // Likewise the questions the documents didn't answer (QUERY_LOG_ENABLED, full key).
  const unansweredPanelAvailable = Boolean(config?.query_log_enabled) && !readOnly;
  useEffect(() => {
    if (rail === 'unanswered' && !unansweredPanelAvailable) setRail('chat');
  }, [rail, unansweredPanelAvailable]);
  const hiddenRailPanels = useMemo(
    () => hiddenPanels(feedbackPanelAvailable, unansweredPanelAvailable),
    [feedbackPanelAvailable, unansweredPanelAvailable],
  );

  // A trace row is a debugging artifact, so opening one implies Engineer mode — the
  // Trace tab does not exist in Reader and the click would otherwise appear to do
  // nothing. Pins the inspector to that trace rather than the newest turn.
  const handleSelectTrace = useStableHandler((traceId: string) => {
    setPinnedTraceId(traceId);
    setMode('engineer');
    setInspectorTab('trace');
    // An explicit request to see the inspector: it outranks the width rule the same way
    // a manual toggle does.
    showInspector();
    // On a narrow screen the inspector opens as a drawer; the panel drawer the trace was
    // picked from would otherwise sit on top of it.
    setPanelDrawerOpen(false);
  });

  // Widening past the breakpoint turns the drawer back into a column; forget that it was
  // open so narrowing again later doesn't pop it up unasked.
  useEffect(() => {
    if (!panelCollapsed) setPanelDrawerOpen(false);
  }, [panelCollapsed]);

  // Rail click: on narrow screens it also opens the drawer, and clicking the panel that
  // is already showing closes it again (the rail is the drawer's toggle).
  const handleRailSelect = useStableHandler((panel: RailPanel) => {
    if (panelCollapsed) setPanelDrawerOpen((open) => !(open && panel === rail));
    setRail(panel);
  });

  const apiReachable = apiStatus === 'ok';

  // Files dropped anywhere outside the corpus panel's own dropzone upload directly
  // (that dropzone stages them for review instead).
  // A dropped folder: its documents only (one per name, at most MAX_FOLDER_FILES),
  // with a note of what was left out rather than one rejection per stray file.
  const uploadFolder = useStableHandler((files: File[]) => {
    const pick = pickFolderFiles(files, acceptedExtensions(config?.supported_extensions));
    const note = folderPickNote(pick);
    const maxBytes = config?.max_upload_bytes ?? DEFAULT_MAX_UPLOAD_BYTES;
    const { accepted, rejected } = validateUploads(
      pick.files,
      maxBytes,
      acceptedExtensions(config?.supported_extensions),
    );
    if (note || rejected.length > 0) {
      pushToast({
        tone: 'warn',
        title: `${accepted.length} file${accepted.length === 1 ? '' : 's'} from the folder uploading`,
        body: [note, ...rejected.map((item) => item.message)].filter(Boolean).join(' '),
      });
    }
    if (accepted.length > 0) {
      showPanel('corpus');
      void handleUpload(accepted);
    }
  });
  const fileDragActive = useWindowFileDrop(apiReachable && !readOnly, (files) => {
    const { accepted, rejected } = validateUploads(
      files,
      config?.max_upload_bytes ?? DEFAULT_MAX_UPLOAD_BYTES,
      acceptedExtensions(config?.supported_extensions),
    );
    if (rejected.length > 0) {
      pushToast({
        tone: 'warn',
        title: `${rejected.length} file${rejected.length === 1 ? '' : 's'} not uploaded`,
        body: rejected.map((item) => item.message).join(' '),
      });
    }
    if (accepted.length > 0) {
      showPanel('corpus');
      void handleUpload(accepted);
    }
  }, uploadFolder);
  const booted = apiStatus === 'ok' || apiStatus === 'unauthorized';
  const {
    connectionLost,
    version: serverVersion,
    updatedTo,
    indexProblem,
    failedModels,
    servicesDown,
    readiness,
  } = useServerHealth(booted);
  useCorpusAutoRefresh(
    corpusChecked && apiStatus === 'ok',
    connectionLost,
    useStableHandler(() => refreshDocuments().catch(noteAuthFailure)),
    indexedFilenames.length === 0
      ? EMPTY_CORPUS_POLL_MS
      : config?.watch_enabled
        ? WATCHED_CORPUS_POLL_MS
        : null,
  );

  const reloadPage = useStableHandler(() => window.location.reload());
  const noDocs = corpusChecked && !listingFailed && indexedFilenames.length === 0;
  const booting = apiStatus === 'checking' || !corpusChecked;
  // Auth takes precedence over noDocs: when the corpus list 401s, "no documents" is a
  // symptom, and telling the user to upload one would send them at a call that 401s too.
  const composerHint =
    apiStatus === 'unauthorized'
      ? 'This server requires an API key — add one in Settings.'
      : connectionLost
        ? 'Can’t reach the server right now — checking again every few seconds.'
        : listingFailed
          ? 'Couldn’t load the document list — retrying. You can still ask.'
          : noDocs
            ? 'Upload a document to start.'
            : booting
              ? 'Connecting…'
              : undefined;
  const activeConversation = conversations.find((c) => c.id === activeConversationId);
  const streamingTitle = conversations.find((c) => c.id === streamingConversationId)?.title;
  const streamingElsewhereHint = `Still answering in “${streamingTitle ?? 'another conversation'}” — you can type, and send once it finishes.`;
  // From the conversation asked in, which needn't be the one open by the time it ends.
  const lastQuestion = [...askedTurns].reverse().find((turn) => turn.role === 'user')?.content;
  const lastTurn = askedTurns[askedTurns.length - 1];
  // A question stopped before the server sent anything leaves no answer turn at all: the
  // question itself is the last turn, and it read as "Answer ready".
  const lastOutcome =
    lastTurn?.role === 'error'
      ? 'failed'
      : lastTurn?.stopped || lastTurn?.role === 'user'
        ? 'stopped'
        : 'answered';
  useBackgroundNotice(pending, notifyOnAnswer, lastQuestion, lastOutcome);
  // The selected provider's model. Each answer records the one it was asked of
  // (ChatTurn.model); this labels answers saved before that, and the composer.
  const currentModelLabel = config
    ? selectedProvider === 'ollama'
      ? config.llm_model
      : selectedProvider === 'openai_compatible'
        ? config.openai_compatible_model || 'OpenAI-compatible'
        : selectedProvider === 'anthropic'
          ? config.anthropic_model || 'Anthropic'
          : config.openai_model
    : undefined;

  // ...or the passages a search finds for it, without asking the model again.
  const searchForPassages = useStableHandler((question: string) => {
    setSearchRequest({ query: question, id: Date.now() });
    setRail('search');
    if (panelCollapsed) setPanelDrawerOpen(true);
  });
  // ...or the passages most like one the viewer shows ("more like this").
  const findSimilarPassages = useStableHandler((chunkId: string, label: string) => {
    setSourceView(null);
    setSearchRequest({ like: { chunkId, label }, id: Date.now() });
    setRail('search');
    if (panelCollapsed) setPanelDrawerOpen(true);
  });
  // Like the composer's Send: not before the app has booted, nor with a refused key —
  // and not while any answer streams (one question at a time; it would be ignored).
  const summarizeDisabled = booting || apiStatus === 'unauthorized' || pending;
  const {
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
  } = useAskActions(
    {
      provider: selectedProvider,
      overrides: advancedOptions,
      scopeFilenames,
      selectedTags,
      scopePages,
      modelLabel: currentModelLabel,
    },
    {
      ask,
      editAndResend,
      setScope,
      pushToast,
      panelCollapsed,
      setPanelDrawerOpen,
      closeSourceView: () => setSourceView(null),
      setPinnedTraceId,
      summarizeDisabled,
    },
  );

  // Same reasoning as askQuestion above — passed to both ChatThread and Inspector.
  const handleOpenSource = useCallback(
    (filename: string, chunkId: string, highlights?: string[]) =>
      setSourceView({ filename, chunkId, highlights }),
    [],
  );
  const handleReadDocument = useCallback((filename: string) => {
    setSourceView({ filename, chunkId: null });
    // On a narrow screen the panel is a drawer over the viewer's opener.
    setPanelDrawerOpen(false);
  }, []);
  // Shown at once (ChatMessage's feedbackGiven); taken back if it couldn't be sent, so
  // the buttons don't claim a rating the server never recorded and it can be retried.
  const handleFeedback = useCallback((payload: FeedbackPayload) => {
    const citation = payload.citationSourceNumber;
    // One source rated, or the whole answer: each kept on the turn so it isn't sent twice.
    const mark = (on: boolean) =>
      citation !== undefined
        ? setSourceIrrelevant(payload.turnId, citation, on)
        : setTurnFeedback(payload.turnId, on ? payload.rating : undefined);
    mark(true);
    void api
      .submitFeedback({
        trace_id: payload.traceId,
        question: payload.question,
        answer_excerpt: sliceWhole(payload.answerExcerpt, 2000),
        cited_filenames: payload.citedFilenames,
        rating: payload.rating,
        citation_source_number: citation ?? null,
        kind: payload.kind,
      })
      .catch((err) => {
        mark(false);
        if (noteAuthFailure(err)) return;
        pushToast({
          tone: 'warn',
          title: "Couldn't send your rating",
          body: apiErrorMessage(err, 'Try again in a moment.'),
        });
      });
    // noteAuthFailure only calls state setters; pushToast is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setTurnFeedback, setSourceIrrelevant, pushToast]);

  // Persistence problems arrive as state flags, not events, so they are surfaced with
  // stable ids — a re-render must refresh the same notice rather than stack duplicates.
  useEffect(() => {
    if (persistError) {
      pushToast({
        id: 'persist-error',
        tone: 'bad',
        title: "Chat history couldn't be saved",
        body: "Storage may be full — this session won't be here after a reload.",
        sticky: true,
      });
    } else if (persistPartial) {
      pushToast({
        id: 'persist-partial',
        tone: 'warn',
        title: 'Chat history was only partly saved',
        body: 'Browser storage is full or unavailable. Export anything you need to keep.',
        sticky: true,
      });
    }
    // Sticky, so they must be withdrawn once a later save fully succeeds.
    if (!persistError) dismissToast('persist-error');
    if (!persistPartial) dismissToast('persist-partial');
  }, [persistError, persistPartial, pushToast, dismissToast]);

  const {
    exportMarkdown,
    exportHtml,
    exportJson,
    exportTraces,
    exportingTraces,
    exportAllConversations,
    exportSavedAnswers,
    exportCitedSources,
  } = useConversationExports(
    turns,
    activeConversation?.title,
    savedAnswers,
    allConversationsJson,
    pushToast,
    noteAuthFailure,
  );
  // The print stylesheet (global.css) leaves only the conversation on the page.
  const printConversation = useStableHandler(() => window.print());

  const copyLastAnswer = useStableHandler(() => {
    const answer = [...turns].reverse().find((turn) => turn.role === 'assistant' && turn.content);
    if (!answer) return;
    const done = (ok: boolean) =>
      pushToast(
        ok
          ? { tone: 'good', title: 'Answer copied' }
          : { tone: 'warn', title: "Couldn't copy", body: 'Use the copy button on the answer.' },
      );
    // As the answer's own Copy button does: with its sources as footnotes (the [n]
    // markers dangled), and on plain HTTP too.
    void copyText(withCitationFootnotes(answer.content, answer.sources)).then(done);
  });
  // The conversation list's order (pinned first, its filters and search applied) while
  // it is shown: ] and [ walk what the reader sees. Else pinned first, by recent use.
  const listedOrderRef = useRef<string[] | null>(null);
  const noteListedOrder = useCallback((ids: string[]) => {
    listedOrderRef.current = ids;
  }, []);
  useEffect(() => {
    if (rail !== 'chat') listedOrderRef.current = null;
  }, [rail]);
  const cycleConversation = useStableHandler((step: 1 | -1) => {
    const order =
      listedOrderRef.current ??
      [
        ...conversations.filter((conversation) => conversation.pinned),
        ...conversations.filter((conversation) => !conversation.pinned),
      ].map((conversation) => conversation.id);
    if (order.length < 2) return;
    const index = order.indexOf(activeConversationId ?? '');
    // Not among those listed (filtered out): the first (or last) one.
    const nextIndex = index < 0 ? (step === 1 ? 0 : order.length - 1) : (index + step + order.length) % order.length;
    const next = order[nextIndex];
    if (next && next !== activeConversationId) switchConversation(next);
  });
  // 1-9: that source of the answer that has focus (j/k move it), else of the latest.
  const openAnswerSource = useStableHandler((number: number) => {
    const answers = turns.filter((turn) => turn.role === 'assistant' && turn.content);
    const focusedId = (document.activeElement as HTMLElement | null)?.closest<HTMLElement>(
      '[role="article"][data-role="assistant"]',
    )?.dataset.turnId;
    const answer = answers.find((turn) => turn.id === focusedId) ?? answers[answers.length - 1];
    const source = answer?.sources.find((candidate) => candidate.source_number === number);
    if (!source) return;
    handleOpenSource(
      source.filename,
      source.chunk_id,
      source.spans?.length ? source.spans.map((span) => span.quote) : undefined,
    );
  });
  const moveBetweenAnswers = useStableHandler((step: 1 | -1) => {
    const answers = [
      ...document.querySelectorAll<HTMLElement>('[role="article"][data-role="assistant"]'),
    ];
    if (answers.length === 0) return;
    const current = answers.findIndex((answer) => answer.contains(document.activeElement));
    let target: HTMLElement | undefined;
    if (current >= 0) {
      target = answers[current + step];
    } else {
      // Nothing focused yet: the first answer in view (j) or the last (k).
      const inView = answers.filter((answer) => {
        const box = answer.getBoundingClientRect();
        return box.bottom > 0 && box.top < window.innerHeight;
      });
      const pool = inView.length > 0 ? inView : answers;
      target = step === 1 ? pool[0] : pool[pool.length - 1];
    }
    if (!target) return;
    target.focus({ preventScroll: true });
    // The CSS reduced-motion rule doesn't reach a scripted smooth scroll.
    const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    target.scrollIntoView({ block: 'start', behavior: still ? 'auto' : 'smooth' });
  });
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const unsaveAnswer = useCallback(
    (conversationId: string, turnId: string) => setTurnSaved(turnId, false, conversationId),
    [setTurnSaved],
  );
  const openPalette = useCallback(() => setPaletteOpen(true), []);
  // Settings needs /config to render. Opened before it arrived, the flag sat set with
  // nothing shown — every ⌘ shortcut then thought a dialog was open — and the modal
  // popped up by itself once the config came in.
  const configLoaded = config !== null;
  const openSettings = useCallback(() => {
    if (configLoaded) setSettingsOpen(true);
  }, [configLoaded]);
  const closePanelDrawer = useCallback(() => setPanelDrawerOpen(false), []);
  const changeProvider = useCallback((provider: LlmProvider) => {
    setSelectedProvider(provider);
    writeStored(PROVIDER_STORAGE_KEY, provider);
  }, []);
  const importInputRef = useRef<HTMLInputElement>(null);

  const startUpload = useStableHandler(() => {
    // On a narrow screen the panel is a closed drawer: the picked files were staged
    // out of sight, still waiting on an Upload click nobody could see.
    showPanel('corpus');
    // The panel has to render before its hidden file input can be clicked.
    requestAnimationFrame(() => corpusBrowseInputRef.current?.click());
  });
  // Example questions for an empty conversation: about the newest documents, which the
  // user most likely means to ask about. A click puts one in the composer to edit.
  const {
    presets: scopePresets,
    savePreset: saveScopePreset,
    deletePreset: deleteScopePreset,
    applyPreset: applyScopePreset,
  } = useScopePresets(workspace, indexedFilenames, corpusChecked && !listingFailed, setScope);
  const saveCurrentScope = useCallback(
    (name: string) => saveScopePreset(name, scope),
    [saveScopePreset, scope],
  );
  // A cited document deleted or uploaded again after the answer: flagged on its card.
  const sourceChangeOf = useCallback(
    (filename: string, answeredAtMs: number) =>
      sourceChange(filename, answeredAtMs, documents, corpusChecked && !listingFailed),
    [documents, corpusChecked, listingFailed],
  );
  const suggestions = useMemo(() => {
    const newest = [...indexedFilenames]
      .sort((a, b) => (documents[b]?.uploadedAt ?? 0) - (documents[a]?.uploadedAt ?? 0))
      .slice(0, 3);
    return newest.map((filename) => `What are the key points of ${filename}?`);
  }, [indexedFilenames, documents]);
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const suggestQuestion = useCallback(
    (text: string) => setPrefill((prev) => ({ text, nonce: (prev?.nonce ?? 0) + 1 })),
    [],
  );
  // From the Unanswered panel: into the composer (not sent — usually a document has to
  // be added first), on narrow screens closing the drawer it was picked in.
  const askUnansweredAgain = useStableHandler((question: string) => {
    suggestQuestion(question);
    if (panelCollapsed) setPanelDrawerOpen(false);
  });

  const commands = useCommands(
    {
      pending: streamingHere,
      configLoaded: config !== null,
      inspectorOpen,
      mode,
      theme,
      hasTurns: turns.length > 0,
      conversations,
      activeConversationId,
      readOnly,
      restoring,
      exportingBackup,
      scopedFilenames: selectedFilenames,
      answerInFlight: pending,
      summarizeDisabled,
      documents: indexedFilenames,
      hasAnswer: turns.some((turn) => turn.role === 'assistant' && turn.content !== ''),
    },
    {
      newConversation: () => {
        setRail('chat');
        newConversation();
      },
      upload: startUpload,
      openSettings,
      toggleInspector,
      setMode,
      toggleTheme,
      pinConversation,
      exportMarkdown,
      exportJson,
      exportHtml,
      printConversation,
      searchPassages: () => {
        setRail('search');
        if (panelCollapsed) setPanelDrawerOpen(true);
      },
      browseTraces: () => {
        setRail('traces');
        if (panelCollapsed) setPanelDrawerOpen(true);
      },
      showSavedAnswers: () => {
        setRail('saved');
        if (panelCollapsed) setPanelDrawerOpen(true);
      },
      exportSavedAnswers,
      restoreBackup: () => restoreInputRef.current?.click(),
      downloadBackup,
      importConversation: () => importInputRef.current?.click(),
      clear,
      switchConversation,
      summarize: summarizeDocuments,
      readDocument: handleReadDocument,
      copyLastAnswer,
      exportAllConversations,
      cycleConversation,
      moveBetweenAnswers,
      exportCitedSources,
      openAnswerSource,
      showShortcuts: () => {
        setPaletteOpen(false);
        setShortcutsOpen(true);
      },
    },
    {
      otherDialogOpen:
        settingsOpen || sourceView !== null || shortcutsOpen || summaryFilenames !== null,
      togglePalette: () => setPaletteOpen((prev) => !prev),
      closePalette: () => setPaletteOpen(false),
    },
  );

  const handleImportFile = async (file: File) => {
    const parsed = parseConversationFile(await file.text());
    let count = 0;
    let skipped = 0;
    let removed = 0;
    if (parsed.kind === 'many') {
      const items = parsed.items;
      // Asked first: making room removes conversations, which can't be undone.
      const { valid, room } = importPreview(items);
      const makeRoom =
        valid > room &&
        window.confirm(
          `This file has ${valid} conversations and the history has room for ${room} ` +
            `(it keeps ${MAX_CONVERSATIONS}).\n\nOK: remove up to ${valid - room} of the ` +
            'least recently used conversations (never pinned ones, nor ones with saved answers) ' +
            'to import them all.\n' +
            `Cancel: import only the ${room} that fit.`,
        );
      ({ imported: count, skipped, removed } = importConversations(items, { makeRoom }));
    } else if (parsed.kind === 'one' && importConversation(parsed.raw)) {
      count = 1;
    }
    const imported = count > 0;
    pushToast(
      skipped > 0
        ? {
            tone: 'warn',
            title: `${count} of ${count + skipped} conversations imported`,
            body:
              `History keeps ${MAX_CONVERSATIONS} conversations` +
              (removed > 0 ? `; ${removed} older ones were removed. ` : '. ') +
              'Delete some and import the file again for the rest.',
          }
        : removed > 0
          ? {
              tone: 'good',
              title: `${count} conversations imported`,
              body: `${removed} older conversation${removed === 1 ? ' was' : 's were'} removed to make room.`,
            }
        : imported
          ? {
              tone: 'good',
              title: count === 1 ? 'Conversation imported' : `${count} conversations imported`,
              body: file.name,
            }
          : {
              tone: 'bad',
              title: "Couldn't import that file",
              body: 'Choose a conversation exported as JSON from DocRAG.',
            },
    );
    if (imported) {
      setPinnedTraceId(null);
      setRail('chat');
    }
  };

  // Stable, so the memoized viewer isn't re-rendered (every passage of a long document)
  // on each frame of an answer streaming meanwhile.
  const closeSourceView = useStableHandler(() => setSourceView(null));
  const downloadViewedOriginal = useStableHandler(() => {
    if (sourceView) downloadOriginal(sourceView.filename);
  });

  const handlePanelAction = useStableHandler(() => {
    if (rail === 'chat') {
      newConversation();
      setPanelDrawerOpen(false);
    } else if (rail === 'corpus') {
      if (readOnly) void refreshDocuments().catch(noteAuthFailure);
      else corpusBrowseInputRef.current?.click();
    }
    else if (rail === 'search') setSearchResetKey((key) => key + 1);
    else if (rail === 'feedback') loadFeedback();
    else if (rail === 'unanswered') loadUnanswered();
    else if (rail === 'saved') exportSavedAnswers();
    else loadTraces();
  });

  return (
    <div className="app-shell">
      {/* First stop for Tab: past the rail and the whole context panel to the question. */}
      <a
        className="app-skip-link"
        href={`#${QUESTION_INPUT_ID}`}
        onClick={(event) => {
          event.preventDefault();
          document.getElementById(QUESTION_INPUT_ID)?.focus();
        }}
      >
        Skip to question
      </a>
      {fileDragActive && (
        <div className="app-drop-overlay" data-testid="app-drop-overlay">
          <p>Drop to upload</p>
        </div>
      )}
      <input
        ref={importInputRef}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) void handleImportFile(file);
        }}
        data-testid="import-conversation-input"
      />
      <input
        ref={restoreInputRef}
        type="file"
        accept=".zip,application/zip"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) void restoreBackup(file);
        }}
        data-testid="restore-backup-input"
      />
      <IconRail
        active={rail}
        onSelect={handleRailSelect}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenSettings={openSettings}
        settingsAvailable={configLoaded}
        hidden={hiddenRailPanels}
      />
      <ContextPanel
        title={PANEL_META[rail].title}
        actionLabel={rail === 'corpus' && readOnly ? 'Refresh' : PANEL_META[rail].actionLabel}
        onAction={handlePanelAction}
        apiStatus={connectionLost && booted ? 'error' : apiStatus}
        chunkTotal={chunkTotal}
        drawer={panelCollapsed}
        drawerOpen={panelDrawerOpen}
        onCloseDrawer={closePanelDrawer}
      >
        {rail === 'chat' && (
          <ConversationList
            conversations={conversations}
            activeId={activeConversationId}
            onSwitch={switchConversationFromList}
            onRename={renameConversation}
            onDelete={deleteConversation}
            onPin={pinConversation}
            onSearch={searchConversations}
            streamingId={pending ? streamingConversationId : null}
            onVisibleOrder={noteListedOrder}
          />
        )}
        {rail === 'corpus' && workspacesEnabled && (
          <WorkspaceBar
            current={workspace}
            workspaces={workspaces}
            disabled={pending}
            readOnly={readOnly}
            onSwitch={chooseWorkspace}
            onCreate={createWorkspace}
            onDelete={deleteWorkspace}
          />
        )}
        {rail === 'corpus' && apiStatus === 'ok' && (
          <ServerActivity
            key={`activity-${workspace}`}
            workspace={workspace}
            followed={followedUploads}
            watchEnabled={Boolean(config?.watch_enabled)}
            canScan={!readOnly}
            onJobsFinished={relistAfterOtherJobs}
          />
        )}
        {rail === 'corpus' && (
          <CorpusPanel
            // A fresh panel per workspace: a selection, a pending confirm or a bulk
            // action of the previous one can't act on this one's documents.
            key={workspace}
            filenames={indexedFilenames}
            documents={documents}
            uploads={workspaceUploads}
            onUpload={handleUpload}
            uploadTags={uploadTags}
            onUploadTagsChange={readOnly ? undefined : setUploadTags}
            onDelete={here.deleteDocument}
            maxUploadBytes={config?.max_upload_bytes}
            supportedExtensions={config?.supported_extensions}
            // Only while this conversation's answer streams: an answer in another
            // conversation locked tagging, deleting and uploading for its minute or two.
            disabled={streamingHere}
            browseInputRef={corpusBrowseInputRef}
            onRetryUpload={handleRetryUpload}
            onDismissUpload={handleDismissUpload}
            onCancelUpload={readOnly ? undefined : handleCancelUpload}
            onReindex={config?.raw_documents_enabled && !readOnly ? here.reindex : undefined}
            onReindexMany={config?.raw_documents_enabled && !readOnly ? here.reindexMany : undefined}
            onReindexAllStale={
              config?.raw_documents_enabled && !readOnly ? handleReindexAllStale : undefined
            }
            onSetTags={readOnly ? undefined : here.setTags}
            readOnly={readOnly}
            onOpenOriginal={config?.raw_documents_enabled ? downloadOriginal : undefined}
            onRead={handleReadDocument}
            onUploadFromUrl={config?.url_ingest_enabled && !readOnly ? handleUploadFromUrl : undefined}
            canPreview={!readOnly}
            onSummarize={summarizeDocuments}
            onAskAbout={askAbout}
            summaryMaxDocuments={config?.summary_max_documents}
            onRefreshFromUrl={config?.url_ingest_enabled && !readOnly ? here.refresh : undefined}
            onSetRefreshInterval={readOnly ? undefined : here.setRefreshInterval}
            onRestoreVersion={
              config?.document_versions_enabled && !readOnly ? here.restoreVersion : undefined
            }
            workspace={workspace}
            summarizeDisabled={summarizeDisabled}
            findAnswersCiting={answersCiting}
            queryLogEnabled={Boolean(config?.query_log_enabled) && !readOnly}
            onOpenAnswer={switchConversationFromList}
          />
        )}
        {/* Kept once opened, hidden while another panel shows: its query and results
            used to be lost on every switch to the corpus or traces and back. */}
        {searchOpened && (
          <div className="app-kept-panel" hidden={rail !== 'search'}>
            <SearchPanel
              filenames={selectedFilenames}
              tags={selectedTags}
              narrowing={scopePages}
              availableTags={availableTags}
              onOpenSource={handleOpenSource}
              resetKey={searchResetKey}
              request={searchRequest}
              onAuthFailure={noteAuthFailure}
              onAskAboutPassages={askAboutPassages}
              askDisabled={summarizeDisabled}
            />
          </div>
        )}
        {rail === 'traces' && (
          <TracesPanel
            traces={traces}
            error={tracesError}
            onSelect={handleSelectTrace}
            answerFilter={traceAnswerFilter}
            onAnswerFilter={setTraceAnswerFilter}
            onExport={readOnly ? undefined : exportTraces}
            exporting={exportingTraces}
          />
        )}
        {rail === 'feedback' && (
          <FeedbackPanel
            feedback={feedbackItems}
            error={feedbackError}
            onSelectTrace={handleSelectTrace}
          />
        )}
        {rail === 'saved' && (
          <SavedPanel
            answers={savedAnswers}
            onOpen={switchConversationFromList}
            onUnsave={unsaveAnswer}
            sourceChangeOf={sourceChangeOf}
            workspace={workspace}
          />
        )}
        {rail === 'unanswered' && (
          <UnansweredPanel
            questions={unanswered}
            error={unansweredError}
            onAsk={askUnansweredAgain}
            onSearch={searchForPassages}
            onSelectTrace={handleSelectTrace}
          />
        )}
      </ContextPanel>
      <main className="app-main">
        {apiStatus === 'error' ? (
          <div className="app-main__unreachable" role="alert">
            <p>
              Cannot reach the DocRAG API &mdash; trying again on its own. Start the API service
              if it isn&rsquo;t running.
            </p>
            <button type="button" onClick={retryBoot} data-testid="api-retry">
              Retry now
            </button>
          </div>
        ) : (
          <>
            <ServerNotices
              indexProblem={indexProblem}
              failedModels={failedModels}
              servicesDown={servicesDown}
              llmProvider={readiness?.llm_provider}
              updatedTo={updatedTo}
              workspacesEnabled={workspacesEnabled}
              onReload={reloadPage}
            />
            <ChatHeader
              key={activeConversationId ?? 'none'}
              title={activeConversation?.title ?? 'New chat'}
              scopeLabel={currentScopeLabel}
              mode={mode}
              onSetMode={setMode}
              hasTurns={turns.length > 0}
              onClear={clear}
              onExport={exportMarkdown}
              disabled={streamingHere}
              onOpenPalette={openPalette}
              inspectorOpen={inspectorOpen}
              onToggleInspector={toggleInspector}
            />
            <ChatThread
              // Per conversation: "a new turn arrived" was judged by the turn count, so
              // switching to a conversation of the same length kept the last one's
              // scroll position and unpinned state.
              // (Prefixed: ChatHeader, a sibling, is keyed by the same id.)
              key={`thread-${activeConversationId ?? 'none'}`}
              turns={turns}
              pending={streamingHere}
              busy={pending}
              stage={streamingHere ? stage : null}
              stageTiming={streamingHere ? stageTiming : null}
              onAskEverywhere={askEverywhere}
              onSearchPassages={searchForPassages}
              onToggleSaved={setTurnSaved}
              // Only once the listing is this conversation's workspace's (it switches
              // to it on opening): another's documents would all read "deleted".
              sourceChangeOf={
                conversationWorkspace === null || conversationWorkspace === workspace
                  ? sourceChangeOf
                  : undefined
              }
              engineerMode={mode === 'engineer'}
              currentModelLabel={currentModelLabel}
              onRetry={askQuestion}
              onRegenerate={regenerateQuestion}
              onAskFollowup={askQuestion}
              weakSupportBelow={
                config ? config.rerank_min_score + WEAK_SUPPORT_MARGIN : undefined
              }
              onEditQuestion={editQuestion}
              feedbackEnabled={config?.feedback_enabled}
              onFeedback={handleFeedback}
              onOpenSource={handleOpenSource}
              onCitationHover={handleCitationHover}
              onCitationLeave={handleCitationLeave}
              documentCount={indexedFilenames.length}
              documentsLoading={!corpusChecked}
              suggestions={suggestions}
              onSuggest={suggestQuestion}
              onUpload={readOnly ? undefined : startUpload}
              focusTurn={focusTurn}
            />
            {hoveredCitation && <SourcePreview citation={hoveredCitation} />}
          </>
        )}
        {/* Outside the branch above, at a fixed position among main's children: a toast
            pushed while the API is unreachable used to go nowhere, and the live regions
            remounted (and so were not announced) when it came back. */}
        <ToastRow
          toasts={toasts}
          onDismiss={dismissToast}
          onHold={holdToast}
          onRelease={releaseToast}
        />
        {apiStatus !== 'error' && (
          <Composer
            onSubmit={askQuestion}
            // Typing is fine while the app boots; only sending waits for it.
            disabled={apiStatus === 'unauthorized' || noDocs}
            sendHeld={booting || (pending && !streamingHere)}
            // Only the startup wait queues a send; waiting for another conversation's
            // answer to finish doesn't send this one by itself later.
            sendQueueable={booting && !(pending && !streamingHere)}
            hint={composerHint ?? (pending && !streamingHere ? streamingElsewhereHint : undefined)}
            pending={streamingHere}
            onCancel={cancel}
            scopeLabel={currentScopeLabel}
            indexedFilenames={indexedFilenames}
            selectedFilenames={selectedFilenames}
            onSelectedFilenamesChange={setSelectedFilenames}
            availableTags={availableTags}
            selectedTags={selectedTags}
            pageRange={pageRange}
            onPageRangeChange={setPageRange}
            uploadedAfter={scope.uploadedAfter}
            onUploadedAfterChange={setUploadedAfter}
            uploadedBefore={scope.uploadedBefore}
            onUploadedBeforeChange={setUploadedBefore}
            scopePresets={scopePresets}
            onApplyScopePreset={applyScopePreset}
            onSaveScopePreset={saveCurrentScope}
            onDeleteScopePreset={deleteScopePreset}
            onSelectedTagsChange={setSelectedTags}
            config={config}
            provider={selectedProvider}
            onProviderChange={changeProvider}
            providerLabel={currentModelLabel ?? selectedProvider}
            draftKey={activeConversationId ?? NEW_CONVERSATION_DRAFT}
            prefill={prefill}
          />
        )}
      </main>
      <LazyChunkBoundary
        onDismiss={() => {
          closeInspector();
          setPaletteOpen(false);
          setSettingsOpen(false);
          setSourceView(null);
        }}
      >
        <Suspense fallback={null}>
          {inspectorOpen && apiStatus !== 'error' && (
            <Inspector
              engineerMode={mode === 'engineer'}
              tab={inspectorTab}
              onTabChange={setInspectorTab}
              onClose={closeInspector}
              sources={inspectedTurn?.sources ?? NO_SOURCES}
              traceId={inspectedTraceId ?? null}
              traceReady={inspectedTraceReady}
              onOpenSource={handleOpenSource}
              overlay={tooNarrowForInspector}
              sourcesEmptyMessage={notFoundSourcesMessage(inspectedTurn)}
            />
          )}
          {paletteOpen && (
            <CommandPalette commands={commands} onClose={() => setPaletteOpen(false)} />
          )}
          {shortcutsOpen && (
            <ShortcutsDialog commands={commands} onClose={() => setShortcutsOpen(false)} />
          )}
          {summaryFilenames && (
            <SummaryDialog
              filenames={summaryFilenames}
              onRun={(instruction) => runSummary(summaryFilenames, instruction)}
              onClose={() => setSummaryFilenames(null)}
            />
          )}
          {settingsOpen && config && (
            <SettingsModal
              config={config}
              serverVersion={serverVersion}
              readiness={readiness}
              overrides={advancedOptions}
              onOverridesChange={updateAdvancedOptions}
              onClose={() => setSettingsOpen(false)}
              onSaved={() => {
                pushToast({ tone: 'good', title: 'Settings saved' });
                // A newly entered API key only takes effect on the next request; without
                // this refetch the "API key required" state and empty corpus stayed until
                // a manual reload.
                refreshDocuments().catch((err) => {
                  noteAuthFailure(err);
                });
                // A different key may unlock more (or less), workspaces included.
                void refreshAccess();
                if (workspacesEnabled) void reloadWorkspaces();
              }}
              disabled={pending}
              notifyOnAnswer={notifyOnAnswer}
              onNotifyOnAnswerChange={(value) => {
                setNotifyOnAnswer(value);
                writeStored(NOTIFY_STORAGE_KEY, value ? 'true' : null);
              }}
            />
          )}
          {sourceView && (
            <DocumentViewer
              filename={sourceView.filename}
              chunkId={sourceView.chunkId}
              highlights={sourceView.highlights}
              onClose={closeSourceView}
              onAskAboutPassage={summarizeDisabled ? undefined : askAboutViewedPassage}
              onFindSimilar={findSimilarPassages}
              onDownloadOriginal={
                config?.raw_documents_enabled && documents[sourceView.filename]?.reindexable
                  ? downloadViewedOriginal
                  : undefined
              }
              canPreviewOriginal={canPreviewOriginal(
                sourceView.filename,
                Boolean(config?.raw_documents_enabled),
                documents[sourceView.filename],
              )}
            />
          )}
        </Suspense>
      </LazyChunkBoundary>
    </div>
  );
}

export default App;
