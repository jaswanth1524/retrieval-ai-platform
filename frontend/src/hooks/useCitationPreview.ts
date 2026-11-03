import { useCallback, useEffect, useRef, useState } from 'react';
import type { CitationResponse } from '../api/types';
import { CITATION_PREFETCH_DWELL_MS } from '../appHelpers';
import { prefetchCitationContent } from '../utils/contentPrefetch';

/** The citation the pointer (or focus) rests on, previewed over the answer, and the
 *  viewer page it would open, fetched ahead once it has rested there a moment. */
export function useCitationPreview() {
  const [hoveredCitation, setHoveredCitation] = useState<CitationResponse | null>(null);
  // A citation's viewer page is fetched once the pointer (or focus) rests on it a
  // moment: tabbing through an answer used to request a 61-chunk page per tab stop.
  const prefetchTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(prefetchTimerRef.current), []);
  const handleCitationLeave = useCallback(() => {
    clearTimeout(prefetchTimerRef.current);
    setHoveredCitation(null);
  }, []);
  // Escape dismisses the preview (WCAG 1.4.13: content shown on hover or focus can be
  // dismissed without moving the pointer or focus).
  useEffect(() => {
    if (!hoveredCitation) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setHoveredCitation(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hoveredCitation]);
  // A pointer or focus on a citation usually precedes a click on it: the viewer's code
  // and its first page start loading now, so the click opens it without waiting.
  const handleCitationHover = useCallback((citation: CitationResponse) => {
    setHoveredCitation(citation);
    clearTimeout(prefetchTimerRef.current);
    prefetchTimerRef.current = setTimeout(() => {
      void import('../components/DocumentViewer');
      prefetchCitationContent(citation.filename, citation.chunk_id);
    }, CITATION_PREFETCH_DWELL_MS);
  }, []);

  return { hoveredCitation, setHoveredCitation, handleCitationHover, handleCitationLeave };
}
