/** A word-level difference between two texts, for showing what Regenerate or Reformat
 *  changed in an answer. */

export interface DiffPart {
  kind: 'same' | 'added' | 'removed';
  text: string;
}

/** Past this many words a side, no diff is made (the table is words × words). */
export const MAX_DIFF_WORDS = 1500;

/** The words of ``before`` and ``after`` matched up (longest common subsequence), as
 *  runs of unchanged, removed and added text, whitespace kept with the word before it;
 *  null when either text is longer than ``MAX_DIFF_WORDS``. */
export function wordDiff(before: string, after: string): DiffPart[] | null {
  const a = tokens(before);
  const b = tokens(after);
  if (a.length > MAX_DIFF_WORDS || b.length > MAX_DIFF_WORDS) return null;
  // lengths[i][j]: the longest common run of a[i:] and b[j:], one row at a time.
  const width = b.length + 1;
  const lengths = new Uint16Array((a.length + 1) * width);
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      lengths[i * width + j] =
        word(a[i]) === word(b[j])
          ? lengths[(i + 1) * width + j + 1] + 1
          : Math.max(lengths[(i + 1) * width + j], lengths[i * width + j + 1]);
    }
  }
  const parts: DiffPart[] = [];
  const push = (kind: DiffPart['kind'], text: string) => {
    const last = parts[parts.length - 1];
    if (last && last.kind === kind) last.text += text;
    else parts.push({ kind, text });
  };
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (word(a[i]) === word(b[j])) {
      push('same', b[j]);
      i += 1;
      j += 1;
    } else if (lengths[(i + 1) * width + j] >= lengths[i * width + j + 1]) {
      push('removed', a[i]);
      i += 1;
    } else {
      push('added', b[j]);
      j += 1;
    }
  }
  for (; i < a.length; i += 1) push('removed', a[i]);
  for (; j < b.length; j += 1) push('added', b[j]);
  return parts;
}

/** Each word with the whitespace after it. */
function tokens(text: string): string[] {
  return text.match(/\S+\s*|\s+/g) ?? [];
}

function word(token: string): string {
  return token.trimEnd();
}
