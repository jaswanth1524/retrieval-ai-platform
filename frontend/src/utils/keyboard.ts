import type { KeyboardEvent } from 'react';

/** True while an input method (Chinese, Japanese, Korean...) is composing text.
 *
 * Its Enter confirms the candidate, not the field: acting on it sent the question,
 * renamed the conversation or ran the palette command half-typed. Safari reports the
 * confirming Enter with keyCode 229 rather than isComposing. */
export function isImeComposing(event: KeyboardEvent): boolean {
  return event.nativeEvent.isComposing || event.keyCode === 229;
}
