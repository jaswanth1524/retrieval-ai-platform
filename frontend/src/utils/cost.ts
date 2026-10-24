/** An estimated cost, in dollars: cents and fractions of them both read clearly. */
export function formatCost(usd: number): string {
  if (usd >= 0.01) return `~$${usd.toFixed(2)}`;
  if (usd > 0) return `~$${usd.toPrecision(2)}`;
  return '$0';
}
