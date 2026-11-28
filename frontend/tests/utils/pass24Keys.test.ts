import { describe, expect, it } from 'vitest';
import { ApiClientError } from '../../src/api/errors';
import { describeKey, isReadOnlyRefusal } from '../../src/utils/keyAccess';

describe('named keys (pass 24)', () => {
  it('names the key and the workspaces it may use', () => {
    expect(describeKey({ access: 'full', key_name: 'full', workspaces: null })).toBe(
      'This key works: full access.',
    );
    expect(describeKey({ access: 'read', key_name: 'bob', workspaces: ['legal'] })).toBe(
      'This key (bob) works: read-only access (asking and searching, no changes), in legal only.',
    );
    expect(
      describeKey({ access: 'full', key_name: 'carol', workspaces: ['default', 'hr', 'it'] }),
    ).toBe('This key (carol) works: full access, in the default workspace, hr and it only.');
    expect(describeKey({ access: 'open' })).toBe(
      'The server has no API key set: any key (or none) works.',
    );
  });

  it('tells a read-only refusal from a workspace refusal', () => {
    const readOnly = new ApiClientError(
      'This API key is read-only; changing documents needs the full API key.',
      403,
    );
    const elsewhere = new ApiClientError("This API key can't use the workspace 'legal'.", 403);
    const everywhere = new ApiClientError(
      'This needs an API key that may use every workspace.',
      403,
    );
    expect(isReadOnlyRefusal(readOnly)).toBe(true);
    expect(isReadOnlyRefusal(elsewhere)).toBe(false);
    expect(isReadOnlyRefusal(everywhere)).toBe(false);
  });
});
