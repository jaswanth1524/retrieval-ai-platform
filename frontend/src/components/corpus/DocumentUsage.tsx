import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { QueryLogEntryResponse } from '../../api/types';
import type { SavedAnswer } from '../../hooks/useChat';
import { formatRelativeTime } from '../../utils/relativeTime';

// How many of each list the card shows.
const USAGE_ROWS = 20;

interface DocumentUsageProps {
  filename: string;
  /** Answers in this browser's conversations citing it, newest first. */
  answers: SavedAnswer[];
  /** QUERY_LOG_ENABLED (and the full key): the server's log is asked too. */
  queryLog: boolean;
  onOpenAnswer: (conversationId: string, turnId: string) => void;
  onClose: () => void;
}

/** "Questions that cited this": the answers here that cite the document, and with the
 *  query log on, every logged question whose answer cited it — to see which
 *  documents are relied on and which are never used. */
export default function DocumentUsage({
  filename,
  answers,
  queryLog,
  onOpenAnswer,
  onClose,
}: DocumentUsageProps) {
  const [logged, setLogged] = useState<QueryLogEntryResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!queryLog) return;
    const controller = new AbortController();
    api
      .listQueryLog(200, controller.signal, false, filename)
      .then((response) => setLogged(response.queries))
      .catch((err) => {
        if (!controller.signal.aborted) setError(apiErrorMessage(err, 'Could not read the query log.'));
      });
    return () => controller.abort();
  }, [filename, queryLog]);

  return (
    <div className="corpus-panel__versions" data-testid="corpus-usage">
      <div className="corpus-panel__versions-head">
        <span>Questions that cited this</span>
        <button type="button" onClick={onClose} aria-label={`Close the questions that cited ${filename}`}>
          ✕
        </button>
      </div>
      {answers.length === 0 ? (
        <p>No answer in this browser cites it.</p>
      ) : (
        <ul>
          {answers.slice(0, USAGE_ROWS).map((item) => (
            <li key={`${item.conversationId}:${item.turn.id}`} data-testid="corpus-usage-answer">
              <button
                type="button"
                className="corpus-panel__usage-open"
                onClick={() => onOpenAnswer(item.conversationId, item.turn.id)}
                title={`In “${item.conversationTitle}”`}
                dir="auto"
              >
                {item.question ?? item.conversationTitle}
              </button>
              <span className="mono">{formatRelativeTime(item.turn.timestamp)}</span>
            </li>
          ))}
        </ul>
      )}
      {queryLog && (
        <>
          {error && <p role="alert">{error}</p>}
          {!error && logged === null && <p>Reading the query log…</p>}
          {logged !== null && (
            <p data-testid="corpus-usage-logged">
              {logged.length === 0
                ? 'No logged question cited it.'
                : `${logged.length >= 200 ? '200+' : logged.length} logged question${
                    logged.length === 1 ? '' : 's'
                  } cited it, the latest: “${logged[0].question}”.`}
            </p>
          )}
        </>
      )}
    </div>
  );
}
