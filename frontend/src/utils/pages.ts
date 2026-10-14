/** "p.3", or "p.3–4" for a passage that runs onto the next page (page_end, sent by
 * servers since chunker version 5). */
export function pageLabel(page: number, pageEnd?: number | null): string {
  return pageEnd && pageEnd !== page ? `p.${page}–${pageEnd}` : `p.${page}`;
}
