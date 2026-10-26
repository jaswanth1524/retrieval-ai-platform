/** A day, in the reader's locale ("24 Oct 2026"). */
export const DATE_FORMATTER = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' });

/** Unix seconds as an <input type="date"> value (the local day), or '' for none. */
export function toDateInput(seconds: number | undefined): string {
  if (seconds === undefined) return '';
  const date = new Date(seconds * 1000);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** An <input type="date"> value as Unix seconds at the start of that local day, or
 *  undefined when it is empty or not a date. */
export function fromDateInput(value: string): number | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return undefined;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  const time = date.getTime();
  return Number.isNaN(time) ? undefined : time / 1000;
}
