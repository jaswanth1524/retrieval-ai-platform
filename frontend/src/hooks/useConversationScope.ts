import { useCallback, useMemo } from 'react';
import { scopeLabel } from '../appHelpers';
import { DATE_FORMATTER } from '../utils/dates';
import { scopeNarrowing } from './chatPersistence';
import type { ConversationScope } from './chatPersistence';

type SetScope = (
  scope: ConversationScope | ((prev: ConversationScope) => ConversationScope),
) => void;

/** The open conversation's search scope (see useChat), split into what the composer and
 *  the Search panel edit, and the label the header and composer show for it. */
export function useConversationScope(
  scope: ConversationScope,
  setScope: SetScope,
  indexedFilenames: string[],
) {
  const selectedFilenames = scope.filenames;
  const selectedTags = scope.tags;
  const setSelectedFilenames = useCallback(
    (filenames: string[]) => setScope((prev) => ({ ...prev, filenames })),
    [setScope],
  );
  const setSelectedTags = useCallback(
    (tags: string[]) => setScope((prev) => ({ ...prev, tags })),
    [setScope],
  );
  const pageRange = useMemo(
    () => ({ min: scope.pageMin, max: scope.pageMax }),
    [scope.pageMin, scope.pageMax],
  );
  const setPageRange = useCallback(
    (range: { min?: number; max?: number }) =>
      setScope((prev) => {
        const { pageMin: _min, pageMax: _max, ...rest } = prev;
        return {
          ...rest,
          ...(range.min !== undefined ? { pageMin: range.min } : {}),
          ...(range.max !== undefined ? { pageMax: range.max } : {}),
        };
      }),
    [setScope],
  );
  const setUploadedAfter = useCallback(
    (value: number | undefined) =>
      setScope((prev) => {
        const { uploadedAfter: _after, ...rest } = prev;
        return value !== undefined ? { ...rest, uploadedAfter: value } : rest;
      }),
    [setScope],
  );
  const setUploadedBefore = useCallback(
    (value: number | undefined) =>
      setScope((prev) => {
        const { uploadedBefore: _before, ...rest } = prev;
        return value !== undefined ? { ...rest, uploadedBefore: value } : rest;
      }),
    [setScope],
  );
  // The scope's page range (while exactly one document is scoped) and upload date.
  const scopePages = useMemo(() => scopeNarrowing(scope), [scope]);
  // No selection = search the whole corpus (undefined).
  const scopeFilenames = selectedFilenames.length > 0 ? selectedFilenames : undefined;
  const currentScopeLabel =
    scopeLabel(indexedFilenames, selectedFilenames, selectedTags) +
    (scopePages && (scopePages.page_min !== undefined || scopePages.page_max !== undefined)
      ? ` · p.${scopePages.page_min ?? 1}${scopePages.page_max !== undefined ? `–${scopePages.page_max}` : '+'}`
      : '') +
    (scope.uploadedAfter !== undefined
      ? ` · since ${DATE_FORMATTER.format(scope.uploadedAfter * 1000)}`
      : '') +
    (scope.uploadedBefore !== undefined
      ? ` · up to ${DATE_FORMATTER.format(scope.uploadedBefore * 1000)}`
      : '');

  return {
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
  };
}
