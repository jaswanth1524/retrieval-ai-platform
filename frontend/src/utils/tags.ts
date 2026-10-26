// api/documents.py's MAX_TAGS_PER_DOCUMENT and MAX_TAG_LENGTH.
const MAX_TAGS_PER_DOCUMENT = 20;
const MAX_TAG_LENGTH = 40;

/** "hr, Policy,  hr" as tags: trimmed, case-insensitively once each, at most 20, each
 *  cut to the server's length, as the server would store them. */
export function parseUploadTags(text: string): string[] {
  const seen = new Set<string>();
  const tags: string[] = [];
  for (const raw of text.split(',')) {
    const tag = raw.split(/\s+/).filter(Boolean).join(' ').slice(0, MAX_TAG_LENGTH);
    if (!tag || seen.has(tag.toLowerCase())) continue;
    seen.add(tag.toLowerCase());
    tags.push(tag);
  }
  return tags.slice(0, MAX_TAGS_PER_DOCUMENT);
}
