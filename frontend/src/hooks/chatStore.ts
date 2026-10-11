/** The full chat history in IndexedDB.
 *
 * localStorage holds ~5 MB per origin, and passage text filled it after a few hundred
 * answers — from then on only the active conversation survived a reload. IndexedDB has
 * a quota in the hundreds of megabytes, so the complete history (source text included)
 * lives here. localStorage keeps a compact copy for an instant first paint and for
 * other tabs' `storage` events, which IndexedDB doesn't fire.
 *
 * Every call degrades to a no-op (load resolves null) where IndexedDB is missing or
 * blocked — private windows in some browsers, test environments.
 */

const DB_NAME = 'docrag';
const STORE = 'chat';
const KEY = 'history';

let connection: Promise<IDBDatabase> | null = null;

export function chatStoreAvailable(): boolean {
  try {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  } catch {
    return false;
  }
}

function open(): Promise<IDBDatabase> {
  if (!connection) {
    connection = new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE);
      request.onsuccess = () => {
        const db = request.result;
        // The browser can close the connection under us (WebKit does, and so does
        // clearing site data), and another tab upgrading the database asks us to.
        // Cached as-is, every later transaction threw and every save failed silently.
        db.onclose = () => {
          connection = null;
        };
        db.onversionchange = () => {
          db.close();
          connection = null;
        };
        resolve(db);
      };
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('IndexedDB open blocked'));
    });
    // A failed open is retried next time rather than cached forever.
    connection.catch(() => {
      connection = null;
    });
  }
  return connection;
}

/** Run ``work`` against the database, once more on a fresh connection if the cached
 *  one turns out to be closed (``transaction()`` throws InvalidStateError on it). */
async function withDb<T>(work: (db: IDBDatabase) => Promise<T>): Promise<T> {
  try {
    return await work(await open());
  } catch {
    connection = null;
    return work(await open());
  }
}

export async function loadChatStore(): Promise<unknown> {
  if (!chatStoreAvailable()) return null;
  try {
    return await withDb(
      (db) =>
        new Promise<unknown>((resolve, reject) => {
          const request = db.transaction(STORE, 'readonly').objectStore(STORE).get(KEY);
          request.onsuccess = () => resolve(request.result ?? null);
          request.onerror = () => reject(request.error);
        }),
    );
  } catch {
    return null;
  }
}

function write(apply: (store: IDBObjectStore) => void): Promise<boolean> {
  return withDb(
    (db) =>
      new Promise<boolean>((resolve) => {
        const transaction = db.transaction(STORE, 'readwrite');
        apply(transaction.objectStore(STORE));
        transaction.oncomplete = () => resolve(true);
        transaction.onerror = () => resolve(false);
        transaction.onabort = () => resolve(false);
      }),
  );
}

/** Resolves true once the value is durably written, false when it couldn't be. */
export async function saveChatStore(value: unknown): Promise<boolean> {
  if (!chatStoreAvailable()) return false;
  try {
    return await write((store) => store.put(value, KEY));
  } catch {
    return false;
  }
}

/** Remove the saved history (nothing worth keeping: one empty conversation). */
export async function clearChatStore(): Promise<boolean> {
  if (!chatStoreAvailable()) return false;
  try {
    return await write((store) => store.delete(KEY));
  } catch {
    return false;
  }
}

/** Test seam: drop the cached connection (e.g. after swapping in a fresh fake). */
export function resetChatStoreConnection(): void {
  connection = null;
}
