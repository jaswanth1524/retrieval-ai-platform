import type { ChatTurn } from '../components/ChatMessage';
import type { SourceChange } from './citationFreshness';

/** Why an answer is worth a second look, in a few words each; empty when nothing
 *  stands out. ``sourceChangeOf`` (this workspace's documents only) adds sources
 *  deleted or uploaded again since the answer was written. */
export function reviewReasons(
  turn: ChatTurn,
  sourceChangeOf?: (filename: string, answeredAtMs: number) => SourceChange | null,
): string[] {
  if (turn.role !== 'assistant' || !turn.content) return [];
  const reasons: string[] = [];
  if (turn.notFound) reasons.push('not found in the documents');
  if (turn.weakMatch && !turn.notFound) reasons.push('weak match');
  if (turn.uncited) reasons.push('no sources cited');
  if (turn.autoCited) reasons.push('citations added automatically');
  if (turn.truncated) reasons.push('cut off');
  if (turn.incomplete) reasons.push('interrupted');
  if (sourceChangeOf) {
    const changes = new Set<SourceChange>();
    for (const filename of new Set(turn.sources.map((source) => source.filename))) {
      const change = sourceChangeOf(filename, turn.timestamp);
      if (change) changes.add(change);
    }
    if (changes.has('changed')) reasons.push('a source changed since');
    if (changes.has('deleted')) reasons.push('a source was deleted');
  }
  return reasons;
}

/** How many answers still need a look: with a reason, and not marked reviewed. */
export function unreviewedCount(
  turns: Iterable<ChatTurn>,
  sourceChangeOf?: (filename: string, answeredAtMs: number) => SourceChange | null,
): number {
  let count = 0;
  for (const turn of turns) {
    if (!turn.reviewedAt && reviewReasons(turn, sourceChangeOf).length > 0) count += 1;
  }
  return count;
}
