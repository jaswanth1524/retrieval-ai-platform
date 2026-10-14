import type { PublicConfigResponse } from '../../src/api/types';
import { initialProvider, isProviderUsable } from '../../src/utils/providers';

function config(overrides: Partial<PublicConfigResponse> = {}): PublicConfigResponse {
  return {
    llm_provider: 'ollama',
    ollama_available: true,
    openai_available: false,
    openai_compatible_available: false,
    allowed_request_providers: null,
    ...overrides,
  } as PublicConfigResponse;
}

describe('isProviderUsable', () => {
  it('needs a reachable Ollama, a configured OpenAI and an allowed provider', () => {
    expect(isProviderUsable('ollama', config())).toBe(true);
    expect(isProviderUsable('ollama', config({ ollama_available: false }))).toBe(false);
    expect(isProviderUsable('openai', config())).toBe(false);
    expect(isProviderUsable('openai', config({ openai_available: true }))).toBe(true);
    expect(
      isProviderUsable(
        'openai',
        config({ openai_available: true, allowed_request_providers: ['ollama'] }),
      ),
    ).toBe(false);
    expect(isProviderUsable('anthropic', config())).toBe(false);
    expect(isProviderUsable(null, config())).toBe(false);
  });
});

describe('initialProvider', () => {
  it('keeps a remembered provider that can answer', () => {
    expect(initialProvider('openai', config({ openai_available: true }))).toBe('openai');
  });

  it('falls back from a remembered Ollama that is down to one that can answer', () => {
    // A remembered Ollama used to win even when unreachable, and every question failed.
    const down = config({ ollama_available: false, openai_available: true });
    expect(initialProvider('ollama', down)).toBe('openai');
  });

  it("prefers the server's default over the other usable providers", () => {
    const both = config({
      llm_provider: 'openai_compatible',
      openai_available: true,
      openai_compatible_available: true,
    });
    expect(initialProvider(null, both)).toBe('openai_compatible');
  });

  it('returns null when nothing can answer', () => {
    expect(initialProvider('ollama', config({ ollama_available: false }))).toBeNull();
  });
});
