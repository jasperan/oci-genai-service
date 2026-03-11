# oci-genai-service

Production-ready Python toolkit for Oracle Cloud Infrastructure Generative AI Service.

[![PyPI](https://img.shields.io/pypi/v/oci-genai-service)](https://pypi.org/project/oci-genai-service/)
[![Python](https://img.shields.io/pypi/pyversions/oci-genai-service)](https://pypi.org/project/oci-genai-service/)
[![License](https://img.shields.io/badge/License-UPL--1.0-blue.svg)](https://opensource.org/licenses/UPL)

## Features

- **Unified client** that auto-routes between OpenAI-compatible and native OCI APIs
- **Chat, streaming, and vision** with Llama 4, Grok, GPT-OSS, and Cohere models
- **Function calling** with a `@tool` decorator that auto-generates OpenAI-format schemas
- **Text embeddings** via Cohere Embed v3 models (native OCI SDK)
- **Oracle AI Vector Search** integration for production-grade vector storage
- **RAG pipeline** with Docling document loading, recursive chunking, and retrieval-augmented generation
- **Tool-calling agent** with memory, multi-turn sessions, and configurable reasoning loops
- **Guardrails** for content moderation, PII detection, and prompt injection defense
- **Thinking trace extraction** for reasoning models (Grok Code, GPT-OSS)
- **CLI** for interactive chat, model browsing, embeddings, and RAG operations
- **5 auth methods**: User Principal, API Key, Instance Principal, Resource Principal, Session Token

## Quick Install

```bash
pip install oci-genai-service
```

## Quick Start

<!-- one-command-install -->
> **One-command install** — clone, configure, and run in a single step:
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/jasperan/oci-genai-service/main/install.sh | bash
> ```
>
> <details><summary>Advanced options</summary>
>
> Override install location:
> ```bash
> PROJECT_DIR=/opt/myapp curl -fsSL https://raw.githubusercontent.com/jasperan/oci-genai-service/main/install.sh | bash
> ```
>
> Or install manually:
> ```bash
> git clone https://github.com/jasperan/oci-genai-service.git
> cd oci-genai-service
> # See below for setup instructions
> ```
> </details>


### Chat Completion

```python
from oci_genai_service import GenAIClient

client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")
response = client.chat("Explain Oracle AI Vector Search in 3 sentences.")
print(response.text)
```

### Streaming

```python
from oci_genai_service import GenAIClient

client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")

for chunk in client.chat("Write a haiku about databases.", stream=True):
    print(chunk, end="", flush=True)
```

### Vision (Image + Text)

```python
from oci_genai_service import GenAIClient

client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")

response = client.chat(
    "Describe this architecture diagram.",
    model="meta.llama-3.2-90b-vision-instruct",
    images=["diagram.png"],
)
print(response.text)
```

### RAG with Oracle Vector Search

```python
from oci_genai_service import GenAIClient, RAGPipeline, OracleVectorStore

client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")
store = OracleVectorStore(
    dsn="localhost:1521/FREEPDB1",
    user="genai",
    password="genai",
)

pipeline = RAGPipeline(client=client, vector_store=store)
pipeline.ingest("quarterly-report.pdf")

result = pipeline.query("What was revenue in Q4?")
print(result.text)
for source in result.sources:
    print(f"  [{source['score']:.2f}] {source['chunk'][:80]}...")
```

### Agent with Tools

```python
from oci_genai_service import GenAIClient, Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72F and sunny in {city}"

@tool
def get_population(city: str) -> int:
    """Get population of a city."""
    return 1_000_000

client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")
agent = Agent(client=client, tools=[get_weather, get_population])

result = agent.run("What's the weather and population in Austin?")
print(result.text)
print(f"Tool calls: {len(result.tool_calls)}, Steps: {result.steps}")
```

## Authentication

| Method | `auth_type` | When to Use |
|---|---|---|
| User Principal | `user_principal` | Local development with `~/.oci/config` (default) |
| API Key | `api_key` | Simple token-based auth, no OCI config needed |
| Instance Principal | `instance_principal` | Running on OCI compute instances |
| Resource Principal | `resource_principal` | OCI Functions, Data Science jobs |
| Session Token | `session` | Temporary session-based authentication |

### Environment Variables

```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxx"
export OCI_REGION="us-chicago-1"           # default
export OCI_PROFILE="DEFAULT"               # OCI config profile
export OCI_CONFIG_FILE="~/.oci/config"     # OCI config path
export OCI_GENAI_API_KEY="your-api-key"    # triggers api_key auth
```

### Explicit Config

```python
from oci_genai_service import GenAIClient, AuthConfig

# User Principal (default)
client = GenAIClient(compartment_id="ocid1.compartment.oc1..xxx")

# API Key
client = GenAIClient(api_key="your-key", compartment_id="ocid1.compartment.oc1..xxx")

# Instance Principal
config = AuthConfig(auth_type="instance_principal", compartment_id="ocid1.compartment.oc1..xxx")
client = GenAIClient(config=config)
```

## CLI

The `oci-genai` command is installed automatically.

### Interactive Chat

```bash
oci-genai chat
# Model: meta.llama-4-maverick
# You> What is Oracle Cloud?
# Assistant> ...
```

### One-shot Chat

```bash
oci-genai chat "Explain RAG in one paragraph" --model meta.llama-3.3-70b-instruct
```

### Vision

```bash
oci-genai chat "Describe this image" --image photo.jpg --model meta.llama-3.2-90b-vision-instruct
```

### Browse Models

```bash
# List all models
oci-genai models list

# Filter by vendor
oci-genai models list --vendor meta

# Filter by capability
oci-genai models list --capability vision

# Model details
oci-genai models info meta.llama-4-maverick
```

### Embeddings

```bash
oci-genai embed "Oracle AI Vector Search" --model cohere.embed-english-v3.0
```

### RAG

```bash
# Ingest documents
oci-genai rag ingest ./docs/ --dsn localhost:1521/FREEPDB1 --user genai --password genai

# Query
oci-genai rag query "What are the main findings?" --dsn localhost:1521/FREEPDB1
```

## Oracle Vector Search Setup

Start Oracle Database 26ai Free with Docker Compose:

```yaml
# docker-compose.yml
services:
  oracle-db:
    image: gvenzl/oracle-free:26-slim
    ports:
      - "1521:1521"
    environment:
      ORACLE_PASSWORD: genai
      APP_USER: genai
      APP_USER_PASSWORD: genai
    volumes:
      - oracle-data:/opt/oracle/oradata

volumes:
  oracle-data:
```

```bash
docker compose up -d

# Verify it's ready
docker compose logs -f oracle-db  # wait for "DATABASE IS READY TO USE"
```

The `OracleVectorStore` will auto-create the vector table on first use.

## Supported Models

### Chat Models (OpenAI-Compatible API)

| Model ID | Name | Vendor | Capabilities | Context |
|---|---|---|---|---|
| `meta.llama-4-scout` | Llama 4 Scout | Meta | chat, function_calling, streaming | 131K |
| `meta.llama-4-maverick` | Llama 4 Maverick | Meta | chat, function_calling, streaming | 131K |
| `meta.llama-3.3-70b-instruct` | Llama 3.3 70B Instruct | Meta | chat, function_calling, streaming | 131K |
| `meta.llama-3.1-405b-instruct` | Llama 3.1 405B Instruct | Meta | chat, function_calling, streaming | 131K |
| `meta.llama-3.1-70b-instruct` | Llama 3.1 70B Instruct | Meta | chat, function_calling, streaming | 131K |
| `xai.grok-4.1-fast` | Grok 4.1 Fast | xAI | chat, streaming | 131K |
| `xai.grok-4` | Grok 4 | xAI | chat, streaming | 131K |
| `xai.grok-4-fast` | Grok 4 Fast | xAI | chat, streaming | 131K |
| `xai.grok-3` | Grok 3 | xAI | chat, streaming | 131K |
| `xai.grok-3-mini` | Grok 3 Mini | xAI | chat, streaming | 131K |
| `xai.grok-3-mini-fast` | Grok 3 Mini Fast | xAI | chat, streaming | 131K |
| `xai.grok-code-fast-1` | Grok Code Fast 1 | xAI | chat, streaming, thinking | 131K |
| `openai.gpt-oss-120b` | GPT-OSS 120B | OpenAI | chat, streaming, thinking | 131K |
| `openai.gpt-oss-20b` | GPT-OSS 20B | OpenAI | chat, streaming | 131K |

### Vision Models (OpenAI-Compatible API)

| Model ID | Name | Vendor | Capabilities | Context |
|---|---|---|---|---|
| `meta.llama-3.2-90b-vision-instruct` | Llama 3.2 90B Vision | Meta | chat, vision, streaming | 131K |
| `meta.llama-3.2-11b-vision-instruct` | Llama 3.2 11B Vision | Meta | chat, vision, streaming | 131K |

### Chat Models (Native OCI API)

| Model ID | Name | Vendor | Capabilities | Context |
|---|---|---|---|---|
| `cohere.command-a` | Command A | Cohere | chat, streaming | 262K |
| `cohere.command-r-plus` | Command R+ | Cohere | chat, streaming | 131K |
| `cohere.command-r` | Command R | Cohere | chat, streaming | 131K |

### Embedding Models (Native OCI API)

| Model ID | Name | Vendor | Dimensions |
|---|---|---|---|
| `cohere.embed-english-v3.0` | Embed English v3.0 | Cohere | 1024 |
| `cohere.embed-multilingual-v3.0` | Embed Multilingual v3.0 | Cohere | 1024 |
| `cohere.embed-english-light-v3.0` | Embed English Light v3.0 | Cohere | 384 |
| `cohere.embed-multilingual-light-v3.0` | Embed Multilingual Light v3.0 | Cohere | 384 |

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- `chat_basic.py` — simple chat completion
- `chat_streaming.py` — streaming output
- `vision.py` — image analysis with vision models
- `function_calling.py` — tool use with `@tool` decorator
- `embeddings.py` — text embeddings
- `rag_pipeline.py` — full RAG with Oracle Vector Search
- `agent.py` — tool-calling agent with memory
- `guardrails.py` — content moderation and PII detection
- `thinking.py` — reasoning trace extraction

## License

Copyright (c) 2024, 2025 Oracle and/or its affiliates.

Licensed under the Universal Permissive License v1.0 (UPL-1.0). See [LICENSE](LICENSE) for details.
