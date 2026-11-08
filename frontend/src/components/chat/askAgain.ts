import type { DocumentTask, PassageNarrowing } from '../../api/types';

/** Ask a question again: its task (a summary) and its narrowing, when it had them. */
export type AskAgain = (
  question: string,
  task?: DocumentTask,
  narrowing?: PassageNarrowing,
  /** Regenerate: the answer this one replaces (kept as its earlier version). */
  replaces?: string,
) => void;

/** Ask again, naming the task and narrowing only when there are some (a plain
 *  question keeps its one-argument call). */
export function askAgain(
  ask: AskAgain,
  question: string,
  task: DocumentTask | undefined,
  narrowing?: PassageNarrowing,
  replaces?: string,
): void {
  if (replaces) ask(question, task, narrowing, replaces);
  else if (narrowing) ask(question, task, narrowing);
  else if (task) ask(question, task);
  else ask(question);
}
