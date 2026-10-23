/** An API call that failed: the server's own (client-safe) message, its status when it
 *  answered, and its Retry-After when it gave one. */
export class ApiClientError extends Error {
  statusCode?: number;
  /** Seconds from the response's Retry-After header (a 503 the server expects to clear). */
  retryAfterSeconds?: number;

  constructor(message: string, statusCode?: number, retryAfterSeconds?: number) {
    super(message);
    this.name = 'ApiClientError';
    this.statusCode = statusCode;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}
