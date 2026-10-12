import type { Components } from 'react-markdown';
import { unlinkCitations } from '../utils/citations';

export interface MarkdownProps {
  text: string;
  components: Components;
  /** Still streaming: re-parse only the block the next token extends. */
  streaming?: boolean;
}

/** An answer as plain text: what shows if the markdown chunk can't load (a tab left
 *  open across a deploy asks for a file the server no longer has). Better than an error.
 *  Citation links are turned back into markers: there is no renderer to make them buttons. */
function PlainAnswer({ text }: MarkdownProps) {
  return <p className="chat-message__plain">{unlinkCitations(text)}</p>;
}

export default PlainAnswer;
