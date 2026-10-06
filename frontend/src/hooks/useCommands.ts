import { useEffect, useMemo, useRef } from 'react';
import type { ChatMode } from '../components/ChatHeader';
import type { Command } from '../components/CommandPalette';
import { isShortcutModifier } from '../utils/platform';
import type { ConversationSummary } from './chatPersistence';

/** What the command list shows: labels that flip, commands that can't run right now. */
export interface CommandState {
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
}

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
  } = state;

  const commands = useMemo<Command[]>(() => {
    const run = actionsRef;
    return [
      {
        id: 'new-conversation',
        glyph: '＋',
        label: 'New conversation',
        // Not ⌘N: browsers reserve it (new window) and never deliver it to the page.
        shortcut: '⌘⇧O',
        run: () => run.current.newConversation(),
        disabled: pending,
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
        run: () => run.current.toggleInspector(),
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
        id: 'search-passages',
        glyph: '⌕',
        label: 'Search passages',
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
              label: 'Download a backup of all documents (zip)',
              // Building the zip can take a while for a large corpus; without a guard
              // every impatient re-run started another full export on the server.
              run: () => run.current.downloadBackup(),
              disabled: exportingBackup,
            },
          ]),
      {
        id: 'import-json',
        glyph: '⇧',
        label: 'Import a conversation (JSON export)',
        run: () => run.current.importConversation(),
        disabled: pending,
      },
      {
        id: 'clear',
        glyph: '✕',
        label: 'Clear this conversation',
        run: () => run.current.clear(),
        disabled: pending || !hasTurns,
      },
      // Every saved conversation, so the palette doubles as conversation search.
      ...conversations
        .filter((conversation) => conversation.id !== activeConversationId)
        .map((conversation) => ({
          id: `conversation-${conversation.id}`,
          glyph: '☰',
          label: `Open: ${conversation.title}`,
          run: () => run.current.switchConversation(conversation.id),
          disabled: pending,
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
  ]);

  // Read at keypress time so the listener is registered once.
  const commandsRef = useRef(commands);
  commandsRef.current = commands;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isShortcutModifier(event) || event.altKey) return;
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
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  return commands;
}
