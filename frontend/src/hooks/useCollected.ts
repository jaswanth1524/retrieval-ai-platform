import { useEffect, useState } from 'react';
import { getWorkspace } from '../api/client';
import type { SelectedQuote } from '../components/SelectionActions';
import {
  BASKET_STORAGE_KEY,
  QUOTES_STORAGE_KEY,
  addToBasket,
  readBasket,
  readQuotes,
  saveQuote,
  type BasketPassage,
} from '../utils/collected';
import { newId } from '../utils/id';
import type { UseToastsResult } from './useToasts';
import { useStableHandler } from './useStableHandler';

/**
 * Quotes and the passage basket ("Quotes & passages"), kept in this browser
 * (utils/collected.ts) and followed across tabs.
 */
export function useCollected(pushToast: UseToastsResult['push']) {
  const [quotes, setQuotes] = useState(readQuotes);
  const [basket, setBasket] = useState(readBasket);
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === QUOTES_STORAGE_KEY) setQuotes(readQuotes());
      if (event.key === BASKET_STORAGE_KEY) setBasket(readBasket());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const keepQuote = useStableHandler((quote: SelectedQuote) => {
    setQuotes(
      saveQuote({
        id: newId(),
        ...quote,
        workspace: getWorkspace() ?? '',
        savedAt: Date.now(),
      }),
    );
  });
  const collectPassages = useStableHandler(
    (passages: Omit<BasketPassage, 'workspace' | 'addedAt'>[]) => {
      const next = addToBasket(
        passages.map((passage) => ({
          ...passage,
          workspace: getWorkspace() ?? '',
          addedAt: Date.now(),
        })),
      );
      setBasket(next);
      pushToast({ tone: 'good', title: `${passages.length === 1 ? 'Passage' : `${passages.length} passages`} in the basket` });
    },
  );

  return { quotes, setQuotes, basket, setBasket, keepQuote, collectPassages };
}
