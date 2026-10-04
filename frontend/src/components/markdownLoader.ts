import { useEffect, useState } from 'react';
import type { ComponentType } from 'react';
import PlainAnswer, { type MarkdownProps } from './PlainAnswer';

type Renderer = ComponentType<MarkdownProps>;

let loading: Promise<Renderer> | null = null;
let loaded: Renderer | null = null;

/** Loads the markdown renderer (its own chunk) once. The app calls it when idle after
 *  boot, so it is usually there before the first answer arrives. A chunk that fails to
 *  load (a tab left open across a deploy, a network blip) is plain text for now; the
 *  next answer to mount tries again. */
export function loadMarkdown(): Promise<Renderer> {
  loading ??= import('./Markdown').then(
    (module) => {
      loaded = module.default as Renderer;
      return loaded;
    },
    () => {
      loading = null;
      return PlainAnswer;
    },
  );
  return loading;
}

/** The markdown renderer, or null until it has loaded. Once loaded it is returned on the
 *  first render (no fallback flash), unlike React.lazy's per-mount suspension. */
export function useMarkdownRenderer(): Renderer | null {
  const [renderer, setRenderer] = useState<Renderer | null>(() => loaded);
  useEffect(() => {
    if (renderer) return;
    let live = true;
    void loadMarkdown().then((next) => {
      if (live) setRenderer(() => next);
    });
    return () => {
      live = false;
    };
  }, [renderer]);
  return renderer;
}
