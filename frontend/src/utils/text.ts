/** The first ``limit`` UTF-16 code units of ``text``, never ending inside a surrogate pair.
 *
 * `slice` counts code units, so a cut through an emoji (or any character outside the
 * BMP) left a lone high surrogate: not valid Unicode, and sent to the server as such. */
export function sliceWhole(text: string, limit: number): string {
  if (text.length <= limit) return text;
  const code = text.charCodeAt(limit - 1);
  const end = code >= 0xd800 && code <= 0xdbff ? limit - 1 : limit;
  return text.slice(0, Math.max(end, 0));
}
