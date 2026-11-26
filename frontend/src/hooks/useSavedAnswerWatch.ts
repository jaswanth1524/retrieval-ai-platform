import { useEffect, useRef } from 'react';
import { api } from '../api/client';
import type { DocumentMeta } from '../components/corpus/types';
import { recheckRequest } from '../utils/savedCheck';
import type { SavedAnswer } from './useChat';

// Saved answers rechecked after one change (each recheck takes a question slot).
export const SAVED_WATCH_MAX = 10;

interface Options {
  answers: readonly SavedAnswer[];
  documents: Record<string, DocumentMeta>;
  /** The listing is current (read, and not failed): only then is a change real. */
  ready: boolean;
  workspace: string;
  /** Some saved answers citing what changed no longer hold. */
  onNoLongerHold: (count: number, filenames: string[]) => void;
}

/** Saved answers watched: when a document they cite is uploaded again or deleted, they
 *  are rechecked against the documents as they are now (POST /questions/recheck), and
 *  the reader is told how many no longer hold. The Saved panel's Recheck did this one
 *  answer at a time, and only when someone thought to click it. */
export function useSavedAnswerWatch({ answers, documents, ready, workspace, onNoLongerHold }: Options): void {
  const seenRef = useRef<{ workspace: string; at: Map<string, number | undefined> } | null>(null);
  const answersRef = useRef(answers);
  answersRef.current = answers;
  const notifyRef = useRef(onNoLongerHold);
  notifyRef.current = onNoLongerHold;
  const runRef = useRef<AbortController | null>(null);
  useEffect(() => () => runRef.current?.abort(), []);

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
      return;
    }
    for (const [name, at] of seen.at) {
      if (!now.has(name) || now.get(name) !== at) pendingRef.current.add(name);
    }
    // One still being written (some of its passages indexed) would read as changed for
    // the wrong reason: wait for the listing that has it whole.
    if ([...pendingRef.current].some((name) => documents[name]?.incomplete)) return;
    const changed = new Set(pendingRef.current);
    if (changed.size === 0) return;
    const affected = answersRef.current
      .filter(
        (item) =>
          item.workspace === workspace &&
          item.turn.sources.some((source) => changed.has(source.filename)),
      )
      .slice(0, SAVED_WATCH_MAX);
    if (affected.length === 0) {
      for (const name of changed) pendingRef.current.delete(name);
      return;
    }
    runRef.current?.abort();
    const controller = new AbortController();
    runRef.current = controller;
    void (async () => {
      let broken = 0;
      // One at a time: each takes one of the server's question slots.
      for (const item of affected) {
        try {
          const result = await api.recheckAnswer(recheckRequest(item), controller.signal, workspace);
          if (controller.signal.aborted) return;
          // What it found, not ``still_holds``: that is false for any answer longer than
          // one recheck reads (``more``), and every long saved answer was reported.
          const gone = result.sources.some((source) => source.state === 'deleted');
          const unbacked = result.sentences.some(
            (sentence) => sentence.status === 'unsupported' || sentence.status === 'numbers_differ',
          );
          if (gone || unbacked) broken += 1;
        } catch {
          if (controller.signal.aborted) return;
          // A busy or unreachable server: these changes are looked at again with the
          // next listing that changes.
          if (broken > 0) notifyRef.current(broken, [...changed].sort());
          return;
        }
      }
      for (const name of changed) pendingRef.current.delete(name);
      if (broken > 0) notifyRef.current(broken, [...changed].sort());
    })();
  }, [documents, ready, workspace]);
}
