import { useRef, useState } from 'react';

/** The citation preview's element id (SourcePreview): the button whose source it shows
 * points at it with aria-describedby, so a screen reader reads the passage too. */
export const SOURCE_PREVIEW_ID = 'docrag-source-preview';

interface PreviewTriggerOptions {
  onOpen?: () => void;
  onHoverStart?: () => void;
  onHoverEnd?: () => void;
}

/** Props for a button that previews a source on hover or focus and opens it on click.
 * Shared by the citation cards and the inline [n] markers, which used to differ:
 *
 * - A tap fires mouseenter too, and with no mouseleave to follow, the preview stuck open
 *   over the thread on touch screens.
 * - Opening the viewer moves focus away; closing it hands focus back to this button,
 *   which re-opened the preview the user had just moved past.
 */
export function usePreviewTrigger({ onOpen, onHoverStart, onHoverEnd }: PreviewTriggerOptions) {
  const touched = useRef(false);
  const skipNextFocus = useRef(false);
  // Whether this button's preview is the one showing (for aria-describedby).
  const [previewing, setPreviewing] = useState(false);
  const start = () => {
    setPreviewing(true);
    onHoverStart?.();
  };
  const end = () => {
    setPreviewing(false);
    onHoverEnd?.();
  };
  return {
    'aria-describedby': previewing ? SOURCE_PREVIEW_ID : undefined,
    onClick: () => {
      end();
      skipNextFocus.current = true;
      onOpen?.();
    },
    onTouchStart: () => {
      touched.current = true;
    },
    onMouseEnter: () => {
      if (!touched.current) start();
    },
    onMouseLeave: () => {
      touched.current = false;
      end();
    },
    onFocus: () => {
      if (skipNextFocus.current) {
        skipNextFocus.current = false;
        return;
      }
      if (!touched.current) start();
    },
    onBlur: end,
  };
}
