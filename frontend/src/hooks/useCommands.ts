import { useEffect, useMemo, useRef } from 'react';
import type { ChatMode } from '../components/ChatHeader';
import type { Command } from '../components/CommandPalette';
import { isShortcutModifier } from '../utils/platform';
import type { ConversationSummary } from './chatPersistence';

/** What the command list shows: labels that flip, commands that can't run right now. */
export interface CommandState {
  /** An answer is streaming into the open conversation (which can't be cleared). */
  pending: boolean;
  /** /config has answered (Settings needs it). */
  configLoaded: boolean;
  inspectorOpen: boolean;
  mode: ChatMode;
  theme: 'dark' | 'light';
  hasTurns: boolean;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  /** The read-only key: no backup or restore. */
  readOnly: boolean;
  restoring: boolean;
  exportingBackup: boolean;
  /** The documents the conversation's scope names: one can be summarized, two compared. */
  scopedFilenames?: string[];
  /** An answer is streaming anywhere: a summary would have to wait for it. */
  answerInFlight?: boolean;
  /** Summaries can't be asked at all right now (booting, the key was refused). */
  summarizeDisabled?: boolean;
  /** Indexed documents, offered as "Read:" and "Summarize:" entries. */
  documents?: string[];
  /** The open conversation has a finished answer to copy. */
  hasAnswer?: boolean;
}

// Past this many documents the palette lists none of them: two entries each would bury
// the commands, and the corpus panel's filter is the way through a corpus that size.
export const PALETTE_MAX_DOCUMENTS = 300;

/** What the commands do. Read when a command runs, so they needn't be stable. */
export interface CommandActions {
  newConversation: () => void;
  upload: () => void;
  openSettings: () => void;
  toggleInspector: () => void;
  setMode: (mode: ChatMode) => void;
  toggleTheme: () => void;
  exportMarkdown: () => void;
  exportJson: () => void;
  browseTraces: () => void;
  searchPassages: () => void;
  restoreBackup: () => void;
  downloadBackup: () => void;
  importConversation: () => void;
  clear: () => void;
  switchConversation: (id: string) => void;
  pinConversation: (id: string, pinned: boolean) => void;
  summarize?: (filenames: string[]) => void;
  readDocument?: (filename: string) => void;
  copyLastAnswer?: () => void;
  exportAllConversations?: () => void;
  /** The next (1) or previous (-1) conversation in the list. */
  cycleConversation?: (step: 1 | -1) => void;
  showShortcuts?: () => void;
}

interface PaletteControls {
  /** Settings or the document viewer is open: no palette or shortcut on top of it. */
  otherDialogOpen: boolean;
  togglePalette: () => void;
  closePalette: () => void;
}

/** The command palette's list, with every ⌘ shortcut it advertises bound to the same
 *  command — dispatched from the list itself, so what the palette shows and what is
 *  bound can't drift apart. */
export function useCommands(
  state: CommandState,
  actions: CommandActions,
  palette: PaletteControls,
): Command[] {
  const actionsRef = useRef(actions);
  actionsRef.current = actions;
  const paletteRef = useRef(palette);
  paletteRef.current = palette;

  const {
    pending,
    configLoaded,
    inspectorOpen,
    mode,
    theme,
    hasTurns,
    conversations,
    activeConversationId,
    readOnly,
    restoring,
    exportingBackup,
    scopedFilenames,
    answerInFlight = false,
    summarizeDisabled = false,
    documents,
    hasAnswer = false,
  } = state;

  const commands = useMemo<Command[]>(() => {
    const run = actionsRef;
    const activeConversation = conversations.find((c) => c.id === activeConversationId);
    return [
      {
        id: 'new-conversation',
        glyph: '＋',
        label: 'New conversation',
        // Not ⌘N: browsers reserve it (new window) and never deliver it to the page.
        shortcut: '⌘⇧O',
        run: () => run.current.newConversation(),
      },
      // Not offered to the read-only key: the server refuses its uploads (403).
      ...(readOnly
        ? []
        : [
            {
              id: 'upload',
              glyph: '↑',
              label: 'Upload a document',
              shortcut: '⌘U',
              run: () => run.current.upload(),
            },
          ]),
      {
        id: 'settings',
        glyph: '⚙',
        label: 'Open settings',
        shortcut: '⌘,',
        run: () => run.current.openSettings(),
        disabled: !configLoaded,
      },
      {
        id: 'toggle-inspector',
        glyph: '◧',
        label: inspectorOpen ? 'Hide inspector' : 'Show inspector',
        // Not ⌘⇧I or ⌘⌥I: the browsers' developer tools.
        shortcut: '⌘\\',
        run: () => run.current.toggleInspector(),
      },
      {
        id: 'shortcuts',
        glyph: '⌨',
        label: 'Keyboard shortcuts',
        shortcut: '?',
        run: () => run.current.showShortcuts?.(),
      },
      {
        id: 'toggle-mode',
        glyph: '◑',
        label: mode === 'engineer' ? 'Switch to Reader mode' : 'Switch to Engineer mode',
        run: () => run.current.setMode(mode === 'engineer' ? 'reader' : 'engineer'),
      },
      {
        id: 'toggle-theme',
        glyph: theme === 'dark' ? '☀' : '☾',
        label: theme === 'dark' ? 'Use light theme' : 'Use dark theme',
        shortcut: '⌘J',
        run: () => run.current.toggleTheme(),
      },
      {
        id: 'export',
        glyph: '⇩',
        label: 'Export conversation as Markdown',
        run: () => run.current.exportMarkdown(),
        disabled: !hasTurns,
      },
      {
        id: 'export-json',
        glyph: '⇩',
        label: 'Export conversation as JSON',
        run: () => run.current.exportJson(),
        disabled: !hasTurns,
      },
      {
        id: 'export-all',
        glyph: '⇩',
        label: 'Export all conversations (JSON)',
        run: () => run.current.exportAllConversations?.(),
        disabled: conversations.every((conversation) => conversation.turnCount === 0),
      },
      {
        id: 'copy-answer',
        glyph: '⧉',
        label: 'Copy the last answer',
        run: () => run.current.copyLastAnswer?.(),
        disabled: !hasAnswer,
      },
      ...(scopedFilenames?.length === 1 || scopedFilenames?.length === 2
        ? [
            {
              id: 'summarize-scope',
              glyph: '∑',
              label:
                scopedFilenames.length === 1
                  ? `Summarize ${scopedFilenames[0]}`
                  : `Compare ${scopedFilenames[0]} and ${scopedFilenames[1]}`,
              run: () => run.current.summarize?.(scopedFilenames),
              disabled: answerInFlight || summarizeDisabled,
            },
          ]
        : []),
      {
        id: 'search-passages',
        glyph: '⌕',
        label: 'Search passages',
        shortcut: '⌘⇧F',
        run: () => run.current.searchPassages(),
      },
      {
        id: 'traces',
        glyph: '◔',
        label: 'Browse traces',
        run: () => run.current.browseTraces(),
      },
      ...(readOnly
        ? []
        : [
            {
              id: 'restore-backup',
              glyph: '⇧',
              label: 'Restore documents from a backup (zip)',
              run: () => run.current.restoreBackup(),
              disabled: restoring,
            },
            {
              id: 'export-corpus',
              glyph: '⇩',
              // While one downloads, the same command cancels it: the server streams one
              // export at a time, so a stalled one otherwise blocked the next for good.
              label: exportingBackup
                ? 'Cancel the backup download'
                : 'Download a backup of all documents (zip)',
              run: () => run.current.downloadBackup(),
            },
          ]),
      {
        id: 'import-json',
        glyph: '⇧',
        label: 'Import a conversation (JSON export)',
        run: () => run.current.importConversation(),
      },
      ...(activeConversation && hasTurns
        ? [
            {
              id: 'pin-conversation',
              glyph: activeConversation.pinned ? '☆' : '★',
              label: activeConversation.pinned ? 'Unpin this conversation' : 'Pin this conversation',
              run: () =>
                run.current.pinConversation(activeConversation.id, !activeConversation.pinned),
            },
          ]
        : []),
      {
        id: 'clear',
        glyph: '✕',
        label: 'Clear this conversation',
        run: () => run.current.clear(),
        disabled: pending || !hasTurns,
      },
      ...(documents && documents.length <= PALETTE_MAX_DOCUMENTS
        ? documents.flatMap((filename) => [
            {
              id: `read-${filename}`,
              glyph: '☰',
              label: `Read: ${filename}`,
              run: () => run.current.readDocument?.(filename),
            },
            {
              id: `summarize-${filename}`,
              glyph: '∑',
              label: `Summarize: ${filename}`,
              run: () => run.current.summarize?.([filename]),
              disabled: answerInFlight || summarizeDisabled,
            },
          ])
        : []),
      // Every saved conversation, so the palette doubles as conversation search.
      ...conversations
        .filter((conversation) => conversation.id !== activeConversationId)
        .map((conversation) => ({
          id: `conversation-${conversation.id}`,
          glyph: '☰',
          label: `Open: ${conversation.title}`,
          run: () => run.current.switchConversation(conversation.id),
        })),
    ];
  }, [
    pending,
    configLoaded,
    inspectorOpen,
    mode,
    theme,
    hasTurns,
    conversations,
    activeConversationId,
    readOnly,
    restoring,
    exportingBackup,
    scopedFilenames,
    answerInFlight,
    summarizeDisabled,
    documents,
    hasAnswer,
  ]);

  // Read at keypress time so the listener is registered once.
  const commandsRef = useRef(commands);
  commandsRef.current = commands;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isShortcutModifier(event) && !event.ctrlKey && !event.metaKey && !event.altKey) {
        plainKey(event);
        return;
      }
      if (!isShortcutModifier(event) || event.altKey) return;
      // A held key repeats: ⌘J flickered the theme, ⌘K opened and closed the palette.
      if (event.repeat) return;
      // Never stack a second dialog over Settings or the document viewer.
      if (paletteRef.current.otherDialogOpen) return;
      const key = event.key.length === 1 ? event.key.toUpperCase() : event.key;
      if (key === 'K' && !event.shiftKey) {
        event.preventDefault();
        paletteRef.current.togglePalette();
        return;
      }
      // Same notation the palette displays ("⌘U", "⌘⇧O").
      const combo = `⌘${event.shiftKey ? '⇧' : ''}${key}`;
      const command = commandsRef.current.find((candidate) => candidate.shortcut === combo);
      if (!command || command.disabled) return;
      event.preventDefault();
      paletteRef.current.closePalette();
      command.run();
    };
    // Keys with no modifier, only while nothing editable has focus (they are text there):
    // ? the shortcut list, ] and [ the next and previous conversation.
    const plainKey = (event: KeyboardEvent) => {
      if (event.repeat || paletteRef.current.otherDialogOpen) return;
      if (isEditable(event.target)) return;
      if (document.querySelector('[aria-modal="true"]')) return;
      const actions = actionsRef.current;
      if (event.key === '?') {
        event.preventDefault();
        actions.showShortcuts?.();
      } else if (event.key === ']' || event.key === '[') {
        event.preventDefault();
        actions.cycleConversation?.(event.key === ']' ? 1 : -1);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  return commands;
}

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT'
  );
}

/** Keys that work with no modifier (outside text fields), for the shortcut list. */
export const PLAIN_KEYS: Array<[keys: string, what: string]> = [
  ['/', 'Type a question'],
  ['?', 'This list'],
  [']  [', 'Next / previous conversation'],
  ['Esc', 'Stop the answer, or close a dialog'],
];
