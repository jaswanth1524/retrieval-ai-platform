/** Where ``query`` occurs in ``text``, ignoring letter case: [start, end) ranges, not
 *  overlapping, in order. An empty query matches nothing. */
export function matchRanges(text: string, query: string): Array<[number, number]> {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return [];
  const haystack = text.toLocaleLowerCase();
  // Lower-casing can change a string's length (İ → i̇); positions then don't line up.
  if (haystack.length !== text.length) return [];
  const ranges: Array<[number, number]> = [];
  let from = 0;
  for (;;) {
    const start = haystack.indexOf(needle, from);
    if (start < 0) return ranges;
    ranges.push([start, start + needle.length]);
    from = start + needle.length;
  }
}
