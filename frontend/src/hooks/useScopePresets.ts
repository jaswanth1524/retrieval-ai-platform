import { useCallback, useEffect, useState } from 'react';
import type { ConversationScope } from './chatPersistence';
import {
  prunePresets,
  readScopePresets,
  withPreset,
  writeScopePresets,
  type ScopePreset,
} from '../utils/scopePresets';

/** The workspace's saved scopes (the composer's "Save scope as…" and its list). Pruned
 *  against the corpus listing once it has loaded, like a conversation's scope. */
export function useScopePresets(
  workspace: string,
  indexedFilenames: string[],
  listed: boolean,
  setScope: (scope: ConversationScope) => void,
) {
  const [presets, setPresets] = useState<ScopePreset[]>(() => readScopePresets(workspace));
  useEffect(() => setPresets(readScopePresets(workspace)), [workspace]);
  useEffect(() => {
    if (!listed) return;
    setPresets((current) => {
      const pruned = prunePresets(current, new Set(indexedFilenames));
      if (pruned !== current) writeScopePresets(workspace, pruned);
      return pruned;
    });
  }, [indexedFilenames, listed, workspace]);

  const savePreset = useCallback(
    (name: string, scope: ConversationScope) =>
      setPresets((current) => {
        const next = withPreset(current, name, scope);
        if (next !== current) writeScopePresets(workspace, next);
        return next;
      }),
    [workspace],
  );
  const deletePreset = useCallback(
    (name: string) =>
      setPresets((current) => {
        const next = current.filter((preset) => preset.name !== name);
        writeScopePresets(workspace, next);
        return next;
      }),
    [workspace],
  );
  const applyPreset = useCallback((preset: ScopePreset) => setScope(preset.scope), [setScope]);

  return { presets, savePreset, deletePreset, applyPreset };
}
