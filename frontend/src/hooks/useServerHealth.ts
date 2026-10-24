import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import type { ReadinessResponse } from '../api/types';
import { useConnectionMonitor } from './useConnectionMonitor';

export interface IndexProblem {
  detail: string;
  /** Originals are kept (RAW_DOCUMENT_DIR): a backup can rebuild the index. */
  originalsKept: boolean;
}

export interface ServerHealth {
  connectionLost: boolean;
  /** The server's version as first seen this session; null until known. */
  version: string | null;
  /** A different version answered since: the server was updated under this page. */
  updatedTo: string | null;
  /** The default workspace's index was built for another embedding setup. */
  indexProblem: IndexProblem | null;
  /** Models that failed to load at startup ("embedding", "reranker"): questions fail
   *  until the server's background retry loads them. */
  failedModels: string[];
}

/** How often readiness is asked again while a model is down, to see it recover. */
export const FAILED_MODEL_RECHECK_MS = 60_000;

/**
 * The connection monitor plus what /health and /health/ready say beyond "up": the
 * server's version (a change means this page's code may no longer match the API) and
 * whether the index matches the embedding setup (a mismatch used to show only as a
 * 409 on the first question).
 */
export function useServerHealth(booted: boolean): ServerHealth {
  const [version, setVersion] = useState<string | null>(null);
  const [updatedTo, setUpdatedTo] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const baselineRef = useRef<string | null>(null);

  const noteVersion = useCallback((seen: string | undefined) => {
    if (!seen) return; // an older server, or one that doesn't say: unknown, no prompt
    if (baselineRef.current === null) {
      baselineRef.current = seen;
      setVersion(seen);
    } else if (seen !== baselineRef.current) {
      setUpdatedTo(seen);
    }
  }, []);

  const connectionLost = useConnectionMonitor(booted, noteVersion);

  const checkReadiness = useCallback(async (signal?: AbortSignal) => {
    try {
      const result = await api.readiness(signal);
      if (!signal?.aborted) setReadiness(result);
    } catch {
      // Unreachable: the connection monitor reports that.
    }
  }, []);

  useEffect(() => {
    if (!booted) return;
    const controller = new AbortController();
    api
      .health(controller.signal)
      .then((health) => noteVersion(health.version))
      .catch(() => {});
    void checkReadiness(controller.signal);
    return () => controller.abort();
  }, [booted, noteVersion, checkReadiness]);

  // Back after an outage: the server may have restarted with another configuration.
  const wasLost = useRef(false);
  useEffect(() => {
    const restored = wasLost.current && !connectionLost;
    wasLost.current = connectionLost;
    if (booted && restored) void checkReadiness();
  }, [booted, connectionLost, checkReadiness]);

  const models = readiness?.models;
  const failedModels = useMemo(
    () =>
      Object.entries(models ?? {})
        // The tokenizer only falls back to a heuristic count; questions still work.
        .filter(([name, state]) => state === 'failed' && name !== 'tokenizer')
        .map(([name]) => name)
        .sort(),
    [models],
  );
  const anyFailed = failedModels.length > 0;
  useEffect(() => {
    if (!booted || !anyFailed) return;
    const timer = setInterval(() => void checkReadiness(), FAILED_MODEL_RECHECK_MS);
    return () => clearInterval(timer);
  }, [booted, anyFailed, checkReadiness]);

  const indexProblem =
    readiness && readiness.index_compatible === false
      ? {
          detail: readiness.index_detail ?? 'The index was built for another embedding setup.',
          originalsKept: Boolean(readiness.originals_kept),
        }
      : null;
  return { connectionLost, version, updatedTo, indexProblem, failedModels };
}
