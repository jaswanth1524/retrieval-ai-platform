import type { UploadProgress } from './types';

export function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot === -1 ? '' : filename.slice(dot + 1).toUpperCase();
}

export function progressPercent(progress?: UploadProgress): number {
  if (!progress) return 0;
  if (progress.chunksTotal > 0) return Math.round((progress.chunksDone / progress.chunksTotal) * 100);
  return progress.state === 'embedding' ? 50 : 0;
}
