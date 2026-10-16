import type { LlmProvider, PublicConfigResponse } from '../api/types';

const PROVIDERS: readonly LlmProvider[] = ['ollama', 'openai', 'openai_compatible', 'anthropic'];

/** Whether the server lets a question pick `provider` (ALLOWED_REQUEST_PROVIDERS). */
export function isProviderAllowed(provider: string, config: PublicConfigResponse): boolean {
  const allowed = config.allowed_request_providers;
  return !allowed || allowed.includes(provider);
}

/** Whether `provider` can answer right now: allowed per request, and configured or
 * reachable. The one copy of this check: the boot-time pick and the composer's provider
 * menu used to disagree about Ollama, so a remembered Ollama stayed selected while it
 * was down and every question failed. */
export function isProviderUsable(
  provider: string | null,
  config: PublicConfigResponse,
): provider is LlmProvider {
  if (provider === null || !isProviderAllowed(provider, config)) return false;
  if (provider === 'ollama') return config.ollama_available;
  if (provider === 'openai') return config.openai_available;
  if (provider === 'openai_compatible') return Boolean(config.openai_compatible_available);
  if (provider === 'anthropic') return Boolean(config.anthropic_available);
  return false;
}

/** The provider to start on: the one remembered in this browser, else the server's
 * default, else any other that can answer. Null when none can (keep the default, so
 * the composer shows why). */
export function initialProvider(
  remembered: string | null,
  config: PublicConfigResponse,
): LlmProvider | null {
  const candidates = [remembered, config.llm_provider, ...PROVIDERS];
  return candidates.find((provider) => isProviderUsable(provider, config)) ?? null;
}
