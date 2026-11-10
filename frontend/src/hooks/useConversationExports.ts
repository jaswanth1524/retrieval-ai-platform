import { useCallback, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import type { ChatTurn } from '../components/ChatMessage';
import {
  chatToJson,
  chatToMarkdown,
  citedSourcesCsv,
  downloadFile,
  safeFilename,
  savedAnswersToMarkdown,
} from '../utils/exportChat';
import { chatToHtml } from '../utils/exportHtml';
import type { SavedAnswer } from './useChat';
import { useStableHandler } from './useStableHandler';
import type { ToastInput } from './useToasts';

/** The downloads offered from the header and the palette: the open conversation (as
 *  Markdown, HTML or JSON), every conversation, the saved answers, and the traces. */
export function useConversationExports(
  turns: ChatTurn[],
  title: string | undefined,
  savedAnswers: SavedAnswer[],
  allConversationsJson: (label?: string) => string,
  pushToast: (toast: ToastInput) => string,
  noteAuthFailure: (err: unknown) => boolean,
) {
  // Read when clicked or exported rather than closed over: the header's Export and the
  // palette keep one identity while an answer streams in.
  const turnsRef = useRef(turns);
  turnsRef.current = turns;
  const conversationTitleRef = useRef(title);
  conversationTitleRef.current = title;
  const savedAnswersRef = useRef(savedAnswers);
  savedAnswersRef.current = savedAnswers;

  const exportMarkdown = useCallback(() => {
    if (turnsRef.current.length === 0) return;
    downloadFile(
      safeFilename(`docrag-${conversationTitleRef.current ?? 'chat'}.md`),
      'text/markdown',
      chatToMarkdown(turnsRef.current, conversationTitleRef.current),
    );
  }, []);
  const exportHtml = useStableHandler(() => {
    if (turnsRef.current.length === 0) return;
    const name = conversationTitleRef.current ?? 'DocRAG conversation';
    void chatToHtml(turnsRef.current, name).then(
      (html) => downloadFile(safeFilename(`docrag-${name}.html`), 'text/html', html),
      () => pushToast({ tone: 'bad', title: 'Could not export the conversation' }),
    );
  });
  const exportJson = useCallback(() => {
    if (turnsRef.current.length === 0) return;
    downloadFile(
      safeFilename(`docrag-${conversationTitleRef.current ?? 'chat'}.json`),
      'application/json',
      chatToJson(turnsRef.current),
    );
  }, []);
  const exportCitedSources = useCallback(() => {
    if (!turnsRef.current.some((turn) => turn.role === 'assistant' && turn.sources.length > 0)) {
      pushToast({ tone: 'warn', title: 'No cited sources yet', body: 'Ask a question first.' });
      return;
    }
    downloadFile(
      safeFilename(`docrag-${conversationTitleRef.current ?? 'chat'}-sources.csv`),
      'text/csv',
      citedSourcesCsv(turnsRef.current, conversationTitleRef.current),
    );
  }, [pushToast]);
  // Every trace of this workspace as JSON Lines: what the server holds (all of them
  // with TRACE_STORE_PATH), for offline analysis or turning into eval rows.
  const [exportingTraces, setExportingTraces] = useState(false);
  const exportTraces = useStableHandler(() => {
    if (exportingTraces) return;
    setExportingTraces(true);
    api
      .exportTraces()
      .then(({ blob, filename }) => downloadFile(filename, 'application/x-ndjson', blob))
      .catch((err) => {
        if (!noteAuthFailure(err)) {
          pushToast({ tone: 'bad', title: 'Could not export the traces', body: apiErrorMessage(err, 'Try again.') });
        }
      })
      .finally(() => setExportingTraces(false));
  });
  const exportAllConversations = useCallback(
    (label?: string) => {
      const stamp = new Date().toISOString().slice(0, 10);
      const name = label ? `docrag-${safeFilename(label)}-${stamp}.json` : `docrag-conversations-${stamp}.json`;
      downloadFile(name, 'application/json', allConversationsJson(label));
    },
    [allConversationsJson],
  );
  const exportSavedAnswers = useCallback(() => {
    const items = savedAnswersRef.current;
    if (items.length === 0) {
      pushToast({ tone: 'warn', title: 'Nothing saved yet', body: 'Save an answer with ☆ first.' });
      return;
    }
    downloadFile('docrag-saved-answers.md', 'text/markdown', savedAnswersToMarkdown(items));
  }, [pushToast]);

  return {
    exportMarkdown,
    exportHtml,
    exportJson,
    exportTraces,
    exportingTraces,
    exportAllConversations,
    exportSavedAnswers,
    exportCitedSources,
  };
}
