/** The labels shown while answer text streams in: a stage line under text that is still
 *  arriving would say nothing new. Any other label after the first token (checking
 *  citations) is shown under the text, which otherwise sits still for that long. */
export const STREAMING_STAGE_LABELS: ReadonlySet<string> = new Set([
  'generating answer…',
  'writing the summary…',
]);
