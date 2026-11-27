import { useEffect, useRef } from 'react';
import { ApiClientError, api } from '../api/client';
import type { RecheckResponse } from '../api/types';
import type { DocumentMeta } from '../components/corpus/types';
import { noLongerHolds, recheckRequest } from '../utils/savedCheck';
import type { SavedAnswer } from './useChat';

// Saved answers rechecked after one change (each recheck takes a question slot).
export const SAVED_WATCH_MAX = 10;

interface Options {
  answers: readonly SavedAnswer[];
  documents: Record<string, DocumentMeta>;
  /** The listing is current (read, and not failed): only then is a change real. */
  ready: boolean;
  workspace: string;
  /** Some saved answers citing what changed no longer hold. ``unchecked``: how many
   *  more cite it than one change rechecks. */
  onNoLongerHold: (count: number, filenames: string[], unchecked: number) => void;
  /** What the recheck of one answer found, holding or not. */
  onChecked?: (item: SavedAnswer, response: RecheckResponse) => void;
}

/** Saved answers watched: when a document they cite is uploaded again or deleted, they
 *  are rechecked against the documents as they are now (POST /questions/recheck), and
 *  the reader is told how many no longer hold. The Saved panel's Recheck did this one
 *  answer at a time, and only when someone thought to click it. */
export function useSavedAnswerWatch({
  answers,
  documents,
  ready,
  workspace,
  onNoLongerHold,
  onChecked,
}: Options): void {
  const seenRef = useRef<{ workspace: string; at: Map<string, number | undefined> } | null>(null);
  const answersRef = useRef(answers);
  answersRef.current = answers;
  const notifyRef = useRef(onNoLongerHold);
  notifyRef.current = onNoLongerHold;
  const checkedRef = useRef(onChecked);
  checkedRef.current = onChecked;
  // The run going: what stops it, and the documents it rechecks for.
  const runRef = useRef<{ controller: AbortController; names: Set<string> } | null>(null);
  useEffect(() => () => runRef.current?.controller.abort(), []);
  // A server without the route: nothing more is asked of it.
  const unsupportedRef = useRef(false);

  // Changed documents not yet rechecked for: a second change while a run was going
  // aborted it, and the first change's answers were never looked at again.
  const pendingRef = useRef(new Set<string>());

  useEffect(() => {
    if (!ready) return;
    const now = new Map(Object.entries(documents).map(([name, meta]) => [name, meta.uploadedAt]));
    const seen = seenRef.current;
    seenRef.current = { workspace, at: now };
    // The first listing (or another workspace's) is what there is, not a change.
    if (seen === null || seen.workspace !== workspace) {
      pendingRef.current.clear();
      unsupportedRef.current = false;
      // The last workspace's run would report into this one.
      runRef.current?.controller.abort();
      runRef.current = null;
      return;
    }
    if (unsupportedRef.current) return;
    let fresh = false;
    for (const [name, at] of seen.at) {
      if (!now.has(name) || now.get(name) !== at) {
        pendingRef.current.add(name);
        fresh = true;
      }
    }
    // One cut off part way through a write would read as changed for the wrong reason:
    // it waits until it is whole. The others don't wait for it (only a re-index makes
    // it whole, and that may never come).
    const changed = new Set([...pendingRef.current].filter((name) => !documents[name]?.incomplete));
    if (changed.size === 0) return;
    // A listing that changed nothing here (a tag, a job's progress) leaves the run be:
    // each one used to start it over from the first answer.
    const running = runRef.current;
    if (!fresh && running && [...changed].every((name) => running.names.has(name))) return;
    const citing = answersRef.current.filter(
      (item) =>
        item.workspace === workspace && item.turn.sources.some((source) => changed.has(source.filename)),
    );
    const done = () => {
      for (const name of changed) pendingRef.current.delete(name);
    };
    if (citing.length === 0) {
      done();
      return;
    }
    const affected = citing.slice(0, SAVED_WATCH_MAX);
    running?.controller.abort();
    const controller = new AbortController();
    const run = { controller, names: changed };
    runRef.current = run;
    void (async () => {
      let broken = 0;
      try {
        // One at a time: each takes one of the server's question slots.
        for (const item of affected) {
          const result = await api.recheckAnswer(recheckRequest(item), controller.signal, workspace);
          if (controller.signal.aborted) return;
          checkedRef.current?.(item, result);
          if (noLongerHolds(result)) broken += 1;
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        if (err instanceof ApiClientError && (err.statusCode === 404 || err.statusCode === 405)) {
          unsupportedRef.current = true;
          pendingRef.current.clear();
        }
        // A busy or unreachable server: nothing is said of a part of them; these
        // changes are looked at again with the next listing.
        return;
      } finally {
        if (runRef.current === run) runRef.current = null;
      }
      done();
      if (broken > 0) notifyRef.current(broken, [...changed].sort(), citing.length - affected.length);
    })();
  }, [documents, ready, workspace]);
}
