import type { LlmProvider, PublicConfigResponse } from '../api/types';
import './ProviderSelector.css';

interface ProviderSelectorProps {
  config: PublicConfigResponse;
  value: LlmProvider;
  onChange: (provider: LlmProvider) => void;
  disabled?: boolean;
}

function ProviderSelector({ config, value, onChange, disabled }: ProviderSelectorProps) {
  const openaiEnabled = config.openai_available;

  return (
    <div className="provider-selector">
      <label className="provider-selector__label" htmlFor="provider-select">
        Answer with
      </label>
      <select
        id="provider-select"
        className="provider-selector__select"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as LlmProvider)}
        data-testid="provider-select"
      >
        <option value="ollama">Local (Ollama · {config.llm_model})</option>
        <option value="openai" disabled={!openaiEnabled}>
          OpenAI ({config.openai_model})
        </option>
      </select>
      {!openaiEnabled && (
        <span className="provider-selector__hint" data-testid="provider-openai-hint">
          Set OPENAI_API_KEY to enable OpenAI
        </span>
      )}
    </div>
  );
}

export default ProviderSelector;
