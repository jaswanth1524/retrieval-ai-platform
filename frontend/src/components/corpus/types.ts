// Corpus types shared by the panel, its cards and useCorpus.
import type { DocumentIngestResponse, IngestJobState } from '../../api/types';

// Structural details of UploadItem below.
export type UploadItemStatus = 'uploading' | 'success' | 'error';

export interface UploadProgress {
  // 'waiting': the server's indexing queue was full (503); retrying after its Retry-After.
  state: IngestJobState | 'waiting';
  chunksDone: number;
  chunksTotal: number;
}

export interface UploadItem {
  id: string;
  filename: string;
  status: UploadItemStatus;
  progress?: UploadProgress;
  result?: DocumentIngestResponse;
  error?: string;
  /** Kept until the upload succeeds or is dismissed, so a failed one can be retried. */
  file?: File;
  /** A re-index of the stored original rather than an upload — retried the same way. */
  reindex?: boolean;
  /** Fetched by the server from this URL (POST /documents/from-url) — retried the same way. */
  sourceUrl?: string;
  /** The workspace it was sent to ('' or absent = the default): a retry, a cancel and
   *  its job polls go there, whichever is in use by then. */
  workspace?: string;
  /** Cancel was pressed and hasn't settled yet. */
  cancelling?: boolean;
}

/** One indexed document's metadata, from GET /documents. */
export interface DocumentMeta {
  chunks: number;
  // Absent for a document indexed before this metadata was stamped at ingest time —
  // the detail line simply omits it.
  pages?: number;
  bytes?: number;
  /** Seconds since the epoch. */
  uploadedAt?: number;
  /** Indexed by an older chunker. */
  stale: boolean;
  /** Has a stored original the server can re-index from (and serve back). */
  reindexable: boolean;
  tags: string[];
}


/** What one card in the corpus panel shows (an indexed document or an upload). */
export interface DocCard {
  key: string;
  filename: string;
  /** Set for cards backed by an upload (in flight or failed) rather than the index. */
  uploadId?: string;
  canRetry?: boolean;
  cancelling?: boolean;
  status: 'indexed' | 'indexing' | 'failed';
  stale?: boolean;
  canReindex?: boolean;
  hasOriginal?: boolean;
  detail: string;
  pct: number;
  error?: string;
  tags: string[];
  bytes?: number;
  uploadedAt?: number;
}
