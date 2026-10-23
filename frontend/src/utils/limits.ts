// The longest question (or search query) the API accepts: api/schemas.py's
// MAX_QUESTION_CHARS. A Python test reads this file to keep the two equal.
export const MAX_QUESTION_CHARS = 4000;

// The most documents one question or search may be scoped to: api/schemas.py's
// REQUEST_FILENAMES_MAX, kept equal by the same test. One more is a 422.
export const MAX_SCOPE_FILENAMES = 500;

// A valid workspace name: api/workspaces.py's _NAME_RE, kept equal by the same test.
export const WORKSPACE_NAME_PATTERN = /^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$/;
