import type { PassageNarrowing } from '../api/types';
import { fromDateInput } from './dates';

/** The panel's own filters, on top of the conversation's scope. */
export interface SearchFilters {
  section: string;
  exclude: string;
  pageMin: string;
  pageMax: string;
  uploadedAfter: string;
  count: string;
}

export const NO_FILTERS: SearchFilters = {
  section: '',
  exclude: '',
  pageMin: '',
  pageMax: '',
  uploadedAfter: '',
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
  let min = positive(filters.pageMin) ?? scoped?.page_min;
  let max = positive(filters.pageMax) ?? scoped?.page_max;
  if (min !== undefined && max !== undefined && min > max) [min, max] = [max, min];
  if (min !== undefined) narrowing.page_min = min;
  if (max !== undefined) narrowing.page_max = max;
  if (filters.section.trim()) narrowing.section = filters.section.trim();
  const exclude = filters.exclude
    .split(',')
    .map((name) => name.trim())
    .filter(Boolean);
  if (exclude.length > 0) narrowing.exclude_filenames = exclude;
  const after = fromDateInput(filters.uploadedAfter) ?? scoped?.uploaded_after;
  if (after !== undefined) narrowing.uploaded_after = after;
  return narrowing;
}
