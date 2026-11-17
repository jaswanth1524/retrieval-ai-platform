import { useEffect, useId, useRef, useState } from 'react';
import './OverflowMenu.css';

export interface OverflowItem {
  label: string;
  onSelect: () => void;
  disabled?: boolean;
  /** Shown as the item's tooltip (why it is disabled, say). */
  title?: string;
  testId?: string;
}

interface OverflowMenuProps {
  /** What the menu is for, read with the button ("Actions for report.pdf"). */
  label: string;
  items: OverflowItem[];
  testId?: string;
  /** The rail's dark colours (the conversation list) or the page's (corpus cards). */
  tone?: 'rail' | 'page';
}

/** "⋯": a row's actions as a menu of labelled items, for touch screens
 *  (useCompactActions). Arrow keys move between items; Escape, a pick or a press
 *  outside closes it, and focus goes back to the button (an action that opens an
 *  editor moves it on from there). */
export default function OverflowMenu({ label, items, testId, tone = 'page' }: OverflowMenuProps) {
  const [open, setOpen] = useState(false);
  const toggleRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLUListElement | null>(null);
  const menuId = useId();

  const itemButtons = () =>
    [...(menuRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? [])].filter(
      (button) => !button.disabled,
    );

  useEffect(() => {
    if (!open) return;
    itemButtons()[0]?.focus();
    const away = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!menuRef.current?.contains(target) && !toggleRef.current?.contains(target)) setOpen(false);
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  }, [open]);

  const close = () => {
    setOpen(false);
    toggleRef.current?.focus();
  };

  return (
    <span className={`overflow-menu overflow-menu--${tone}`}>
      <button
        ref={toggleRef}
        type="button"
        className="overflow-menu__toggle"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={label}
        onClick={() => setOpen((value) => !value)}
        data-testid={testId}
      >
        ⋯
      </button>
      {open && (
        <ul
          ref={menuRef}
          id={menuId}
          role="menu"
          aria-label={label}
          className="overflow-menu__list"
          onKeyDown={(event) => {
            const buttons = itemButtons();
            const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
            if (event.key === 'Escape') {
              event.preventDefault();
              event.stopPropagation();
              close();
            } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
              event.preventDefault();
              const step = event.key === 'ArrowDown' ? 1 : -1;
              buttons[(at + step + buttons.length) % buttons.length]?.focus();
            } else if (event.key === 'Tab') {
              setOpen(false);
            }
          }}
        >
          {items.map((item) => (
            <li key={item.label} role="none">
              <button
                type="button"
                role="menuitem"
                className="overflow-menu__item"
                disabled={item.disabled}
                title={item.title}
                onClick={() => {
                  close();
                  item.onSelect();
                }}
                data-testid={item.testId}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </span>
  );
}
