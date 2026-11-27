import { useCallback, useState } from 'react';
import type { RecheckResponse } from '../api/types';
import { noLongerHolds, recheckSummary, savedKey } from '../utils/savedCheck';
import type { SavedAnswer } from './useChat';

/** One saved answer the watch found no longer holding: what it judged, and why. */
export interface SavedVerdict {
  answer: string;
  text: string;
}

/** What the saved-answer watch found, by answer, for the Saved panel to show on each:
 *  a toast only said how many. Kept for this page load only. */
export function useSavedVerdicts(): [
  Record<string, SavedVerdict>,
  (item: SavedAnswer, response: RecheckResponse) => void,
] {
  const [verdicts, setVerdicts] = useState<Record<string, SavedVerdict>>({});
  const note = useCallback((item: SavedAnswer, response: RecheckResponse) => {
    const key = savedKey(item);
    setVerdicts((prev) => {
      if (noLongerHolds(response)) {
        return { ...prev, [key]: { answer: item.turn.content, text: recheckSummary(response) } };
      }
      if (!(key in prev)) return prev;
      const { [key]: _cleared, ...rest } = prev;
      return rest;
    });
  }, []);
  return [verdicts, note];
}
