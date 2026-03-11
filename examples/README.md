# OCI GenAI Service Examples

Self-contained examples demonstrating every capability of the oci-genai-service toolkit.

## Prerequisites

```bash
pip install oci-genai-service
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..your_compartment
```

For Oracle Vector Search examples, start the database:
```bash
docker compose up -d
```

## Examples

| # | Category | Example | Description | Needs Credentials? |
|---|----------|---------|-------------|-------------------|
| 01 | Chat | [simple_chat.py](01_chat_basics/simple_chat.py) | Basic chat completion | Yes |
| 02 | Streaming | [stream_chat.py](02_streaming/stream_chat.py) | Stream tokens in real-time | Yes |
| 03 | Vision | [describe_image.py](03_vision/describe_image.py) | Describe images with vision models | Yes |
| 04 | Tools | [single_tool.py](04_function_calling/single_tool.py) | Let the model call Python functions | Yes |
| 05 | Embeddings | [embed_texts.py](05_embeddings/embed_texts.py) | Generate text embeddings | Yes |
| 06 | RAG | [rag_chatbot.py](06_rag_oracle_vectordb/rag_chatbot.py) | Q&A over documents with Oracle Vector Search | Yes + Oracle DB |
| 07 | Agents | [simple_agent.py](07_agents/simple_agent.py) | Agent with tool calling and memory | Yes |
| 08 | Guardrails | [content_moderation.py](08_guardrails/content_moderation.py) | Content moderation and PII detection | No |
| 09 | Thinking | [grok_reasoning.py](09_thinking_traces/grok_reasoning.py) | Extract reasoning traces from models | Yes |
| 10 | Models | [list_models.py](10_model_management/list_models.py) | Browse and inspect available models | No |
