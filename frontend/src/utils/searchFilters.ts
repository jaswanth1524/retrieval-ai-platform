import type { PassageNarrowing } from '../api/types';
import { endOfDayFromDateInput, fromDateInput } from './dates';

/** The panel's own filters, on top of the conversation's scope. */
export interface SearchFilters {
  section: string;
  exclude: string;
  pageMin: string;
  pageMax: string;
  uploadedAfter: string;
  uploadedBefore: string;
  /** One tag to search within, in place of the conversation's tags ('' = theirs). */
  tag: string;
  count: string;
}

export const NO_FILTERS: SearchFilters = {
  section: '',
  exclude: '',
  pageMin: '',
  pageMax: '',
  uploadedAfter: '',
  uploadedBefore: '',
  tag: '',
  count: '',
};

export function positive(value: string, max = 1_000_000): number | undefined {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 1 ? Math.min(parsed, max) : undefined;
}

/** The request's narrowing: the panel's filters, else the conversation's. */
export function searchNarrowing(
  filters: SearchFilters,
  scoped: PassageNarrowing | undefined,
): PassageNarrowing {
  const narrowing: PassageNarrowing = {};
  // One range, from one place: the panel's when it sets either end, else the
  // conversation's. Mixing them ("from 50" here, "to 10" there) searched pages 10-50,
  // outside the range either one meant.
  const ownRange = positive(filters.pageMin) !== undefined || positive(filters.pageMax) !== undefined;
  let min = ownRange ? positive(filters.pageMin) : scoped?.page_min;
  let max = ownRange ? positive(filters.pageMax) : scoped?.page_max;
  if (min !== undefined && max !== undefined && min > max) [min, max] = [max, min];
  if (min !== undefined) narrowing.page_min = min;
  if (max !== undefined) narrowing.page_max = max;
  if (filters.section.trim()) narrowing.section = filters.section.trim();
  const exclude = filters.exclude
    .split(',')
    .map((name) => name.trim())
    .filter(Boolean);
  if (exclude.length > 0) narrowing.exclude_filenames = exclude;
  let after = fromDateInput(filters.uploadedAfter) ?? scoped?.uploaded_after;
  let before = endOfDayFromDateInput(filters.uploadedBefore) ?? scoped?.uploaded_before;
  if (after !== undefined && before !== undefined && after > before) [after, before] = [before, after];
  if (after !== undefined) narrowing.uploaded_after = after;
  if (before !== undefined) narrowing.uploaded_before = before;
  return narrowing;
}

/** The tags a search looks within: the panel's one tag, else the conversation's. */
export function searchTags(filters: SearchFilters, scoped: readonly string[]): string[] {
  return filters.tag ? [filters.tag] : [...scoped];
}
