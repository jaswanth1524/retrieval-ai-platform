/** One CSV cell (RFC 4180): quoted when it holds a comma, quote or line break, and
 *  with a leading ' when it would start a spreadsheet formula (= + - @, or a tab or
 *  carriage return before one) — the rows hold what users typed. */
export function csvCell(value: unknown): string {
  let text = value === null || value === undefined ? '' : String(value);
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`;
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** ``rows`` under ``headers`` as CSV text (CRLF line ends, as RFC 4180 has them). */
export function toCsv(headers: string[], rows: unknown[][]): string {
  return [headers, ...rows].map((row) => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
}

/** A Unix time as an ISO date-time (UTC), for a CSV column. */
export function isoTime(seconds: number): string {
  return new Date(seconds * 1000).toISOString();
}
