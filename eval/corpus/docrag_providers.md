# Generation providers

## Default provider

The default generation provider is a local Ollama server reached through LiteLLM, and the
default local model is llama3.1:8b. Nothing about document upload or indexing needs a
hosted model.

## OpenAI

OpenAI generation is optional and selected per deployment or per question. It needs an
OPENAI_API_KEY; embeddings stay local even when OpenAI generates the answer.

## Self-hosted OpenAI-compatible servers

Any server that speaks the OpenAI chat-completions protocol, such as vLLM, the llama.cpp
server, or LM Studio, can generate answers when its base URL and model name are set.
