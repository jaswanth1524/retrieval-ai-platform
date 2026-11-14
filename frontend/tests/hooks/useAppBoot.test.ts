import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

vi.mock('../../src/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../src/api/client')>();
  return {
    ...actual,
    api: { ...actual.api, config: vi.fn(), access: vi.fn() },
  };
});

import { api } from '../../src/api/client';
import type { PublicConfigResponse } from '../../src/api/types';
import { useAppBoot } from '../../src/hooks/useAppBoot';

describe('useAppBoot', () => {
  it('reads /config again on request without re-running the boot callback', async () => {
    const configMock = vi.mocked(api.config);
    vi.mocked(api.access).mockResolvedValue({ access: 'full' } as never);
    configMock.mockResolvedValueOnce({ ollama_available: false } as PublicConfigResponse);
    configMock.mockResolvedValueOnce({ ollama_available: true } as PublicConfigResponse);
    const onConfig = vi.fn();
    const { result } = renderHook(() =>
      useAppBoot({ refreshDocuments: async () => {}, onConfig }),
    );
    await waitFor(() => expect(result.current.config?.ollama_available).toBe(false));
    await act(async () => {
      await result.current.refreshConfig();
    });
    expect(result.current.config?.ollama_available).toBe(true);
    expect(onConfig).toHaveBeenCalledTimes(1);
  });
});
