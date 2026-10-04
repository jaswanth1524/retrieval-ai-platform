import type { Components } from 'react-markdown';

export interface MarkdownProps {
  text: string;
  components: Components;
  /** Still streaming: re-parse only the block the next token extends. */
  streaming?: boolean;
}

/** An answer as plain text: what shows if the markdown chunk can't load (a tab left
 *  open across a deploy asks for a file the server no longer has). Better than an error. */
function PlainAnswer({ text }: MarkdownProps) {
  return <p className="chat-message__plain">{text}</p>;
}

export default PlainAnswer;
