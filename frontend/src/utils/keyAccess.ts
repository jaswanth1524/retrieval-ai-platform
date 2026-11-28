import type { ApiClientError } from '../api/errors';
import type { AccessResponse } from '../api/types';

/** A 403 for a read-only key (its message says so), not a named key kept out of a
 *  workspace ("This API key can't use the workspace …"). */
export function isReadOnlyRefusal(err: ApiClientError): boolean {
  return !/can't use the workspace|every workspace/i.test(err.message);
}

function workspaceList(names: string[]): string {
  const shown = names.map((name) => (name === 'default' ? 'the default workspace' : name));
  if (shown.length <= 1) return shown.join('');
  return `${shown.slice(0, -1).join(', ')} and ${shown[shown.length - 1]}`;
}

/** Settings → Test key: what the key unlocks, named, and where (a named key may be
 *  limited to some workspaces). */
export function describeKey(answer: AccessResponse): string {
  if (answer.access === 'open') return 'The server has no API key set: any key (or none) works.';
  const named =
    answer.key_name && answer.key_name !== 'full' && answer.key_name !== 'read'
      ? ` (${answer.key_name})`
      : '';
  const what =
    answer.access === 'full'
      ? 'full access'
      : 'read-only access (asking and searching, no changes)';
  const where = answer.workspaces ? `, in ${workspaceList(answer.workspaces)} only` : '';
  return `This key${named} works: ${what}${where}.`;
}
