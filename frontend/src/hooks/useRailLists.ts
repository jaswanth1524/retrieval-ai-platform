import { useState } from 'react';
import { api } from '../api/client';
import type {
  FeedbackItemResponse,
  QueryLogEntryResponse,
  TraceSummaryResponse,
} from '../api/types';
import type { RailPanel } from '../components/IconRail';
import { QUERY_LOG_FETCH_LIMIT, mergeQueryLogs } from '../utils/questions';
import { ANSWER_FILTER_STATUSES, type AnswerFilter } from '../utils/traceFilters';
import { usePanelList } from './usePanelList';

/** The rail panels' server lists — ratings, asked questions, traces — each loaded while
 *  its panel shows and again when ``workspace`` (or the trace filter) changes. */
export function useRailLists(
  rail: RailPanel,
  workspace: string,
  onError: (err: unknown) => void,
) {
  const {
    items: feedbackItems,
    error: feedbackError,
    load: loadFeedback,
  } = usePanelList<FeedbackItemResponse>(
    rail === 'feedback',
    // The server's cap (FEEDBACK_LIST_MAX_LIMIT): the panel filters what it holds.
    async (signal) => (await api.listFeedback(500, signal)).feedback,
    onError,
    'Could not load feedback.',
    workspace,
  );

  const {
    items: unanswered,
    error: unansweredError,
    load: loadUnanswered,
  } = usePanelList<QueryLogEntryResponse>(
    rail === 'unanswered',
    // The latest asks, and the unanswered ones further back: the Unanswered view is
    // the whole log's, not the last window's.
    async (signal) => {
      const [recent, uncovered] = await Promise.all([
        api.listQueryLog(QUERY_LOG_FETCH_LIMIT, signal),
        api.listQueryLog(QUERY_LOG_FETCH_LIMIT, signal, true),
      ]);
      return mergeQueryLogs(recent.queries, uncovered.queries);
    },
    onError,
    'Could not load the asked questions.',
    workspace,
  );

  const [traceAnswerFilter, setTraceAnswerFilter] = useState<AnswerFilter>('any');
  const {
    items: traces,
    error: tracesError,
    load: loadTraces,
  } = usePanelList<TraceSummaryResponse>(
    rail === 'traces',
    async (signal) =>
      (await api.listTraces(signal, ANSWER_FILTER_STATUSES[traceAnswerFilter])).traces,
    onError,
    'Could not load traces.',
    `${workspace}\u0000${traceAnswerFilter}`,
  );

  return {
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
  };
}
