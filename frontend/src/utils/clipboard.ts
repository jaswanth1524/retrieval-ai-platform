/** Copy without the async Clipboard API, which browsers gate behind a secure context.
 *
 *  DocRAG is self-hosted, so plain HTTP on a LAN address is a first-class deployment
 *  — and there `navigator.clipboard` is undefined. Hiding the copy actions there
 *  removed the feature from exactly the users the project targets. */
function legacyCopy(text: string): boolean {
  const area = document.createElement('textarea');
  area.value = text;
  // Keep it out of view and off the tab order, but still selectable.
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.top = '-9999px';
  // select() moves focus to the throwaway field; once it was removed, focus fell to
  // <body> and a keyboard user lost their place. It goes back where it was.
  const previous = document.activeElement;
  document.body.appendChild(area);
  area.select();
  try {
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    document.body.removeChild(area);
    if (previous instanceof HTMLElement) previous.focus();
  }
}

/** Put ``text`` on the clipboard; whether it worked. The Clipboard API when there is
 *  one (a denied write — permissions policy — falls back too), else the legacy way. */
export function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard) {
    return navigator.clipboard.writeText(text).then(
      () => true,
      () => legacyCopy(text),
    );
  }
  return Promise.resolve(legacyCopy(text));
}
