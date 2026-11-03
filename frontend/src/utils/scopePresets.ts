// Named search scopes (files, tags, pages, dates), saved from the composer and picked
// again later. Kept in localStorage per workspace: a scope names that workspace's files.
import { isEmptyScope, sanitizeScope, type ConversationScope } from '../hooks/chatPersistence';
import { readStoredJson, writeStored } from './safeStorage';

export const SCOPE_PRESETS_STORAGE_KEY = 'docrag-scope-presets';
export const MAX_SCOPE_PRESETS = 20;
export const SCOPE_PRESET_NAME_MAX_CHARS = 60;

export interface ScopePreset {
  name: string;
  scope: ConversationScope;
}

type Stored = Record<string, ScopePreset[]>;

function sanitizePresets(value: unknown): ScopePreset[] {
  if (!Array.isArray(value)) return [];
  const presets: ScopePreset[] = [];
  const names = new Set<string>();
  for (const item of value) {
    if (!item || typeof item !== 'object') continue;
    const raw = item as Partial<ScopePreset>;
    const name = typeof raw.name === 'string' ? raw.name.trim().slice(0, SCOPE_PRESET_NAME_MAX_CHARS) : '';
    const scope = sanitizeScope(raw.scope);
    if (!name || !scope || isEmptyScope(scope) || names.has(name)) continue;
    names.add(name);
    presets.push({ name, scope });
    if (presets.length >= MAX_SCOPE_PRESETS) break;
  }
  return presets;
}

function readAll(): Stored {
  const stored = readStoredJson<unknown>(SCOPE_PRESETS_STORAGE_KEY);
  return stored && typeof stored === 'object' && !Array.isArray(stored) ? (stored as Stored) : {};
}

/** ``workspace``'s presets ('' = the default), sanitized. */
export function readScopePresets(workspace: string): ScopePreset[] {
  return sanitizePresets(readAll()[workspace]);
}

export function writeScopePresets(workspace: string, presets: ScopePreset[]): void {
  const all = readAll();
  if (presets.length > 0) all[workspace] = presets.slice(0, MAX_SCOPE_PRESETS);
  else delete all[workspace];
  writeStored(SCOPE_PRESETS_STORAGE_KEY, Object.keys(all).length ? JSON.stringify(all) : null);
}

/** ``presets`` with ``scope`` saved under ``name`` (replacing one of that name, newest
 *  first); unchanged when the name is empty, the scope narrows nothing, or the list is
 *  full. */
export function withPreset(presets: ScopePreset[], name: string, scope: ConversationScope): ScopePreset[] {
  const trimmed = name.trim().slice(0, SCOPE_PRESET_NAME_MAX_CHARS);
  const clean = sanitizeScope(scope);
  if (!trimmed || !clean || isEmptyScope(clean)) return presets;
  const others = presets.filter((preset) => preset.name !== trimmed);
  if (others.length >= MAX_SCOPE_PRESETS) return presets;
  return [{ name: trimmed, scope: clean }, ...others];
}

/** ``presets`` without files no longer indexed — as a conversation's scope is pruned —
 *  dropping a preset left narrowing nothing. The same array when nothing changed. */
export function prunePresets(presets: ScopePreset[], indexed: ReadonlySet<string>): ScopePreset[] {
  let changed = false;
  const kept: ScopePreset[] = [];
  for (const preset of presets) {
    const filenames = preset.scope.filenames.filter((name) => indexed.has(name));
    if (filenames.length === preset.scope.filenames.length) {
      kept.push(preset);
      continue;
    }
    changed = true;
    const scope = { ...preset.scope, filenames };
    // Its page range belongs to the one document it named.
    if (filenames.length !== 1) {
      delete scope.pageMin;
      delete scope.pageMax;
    }
    if (!isEmptyScope(scope)) kept.push({ ...preset, scope });
  }
  return changed ? kept : presets;
}
