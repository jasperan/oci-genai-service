"""CLI entry point for oci-genai-service."""

from __future__ import annotations

import sys
import click

from oci_genai_service import __version__


@click.group()
@click.version_option(version=__version__, prog_name="oci-genai")
def cli():
    """OCI GenAI Service — CLI toolkit for Oracle Cloud Generative AI."""
    pass


@cli.group()
def models():
    """Browse and inspect available models."""
    pass


@models.command("list")
@click.option("--vendor", help="Filter by vendor (meta, cohere, xai, openai)")
@click.option("--capability", help="Filter by capability (chat, vision, embedding, function_calling)")
def models_list(vendor, capability):
    """List available OCI GenAI models."""
    from oci_genai_service.models import list_models as _list_models

    results = _list_models(vendor=vendor, capability=capability)
    if not results:
        click.echo("No models match the given filters.")
        return

    click.echo(f"{'Model ID':<45} {'Name':<30} {'API':<20} {'Capabilities'}")
    click.echo("-" * 120)
    for m in results:
        caps = ", ".join(m.capabilities)
        click.echo(f"{m.id:<45} {m.name:<30} {m.api:<20} {caps}")


@models.command("info")
@click.argument("model_id")
def models_info(model_id):
    """Show detailed info about a model."""
    from oci_genai_service.models import get_model

    try:
        m = get_model(model_id)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    click.echo(f"Model: {m.name}")
    click.echo(f"ID: {m.id}")
    click.echo(f"Vendor: {m.vendor}")
    click.echo(f"API: {m.api}")
    click.echo(f"Context Window: {m.context_window}")
    click.echo(f"Capabilities: {', '.join(m.capabilities)}")
    click.echo(f"Regions: {', '.join(m.regions)}")
    if m.embedding_dims:
        click.echo(f"Embedding Dims: {m.embedding_dims}")


@cli.command()
@click.argument("prompt", required=False)
@click.option("--model", "-m", default="meta.llama-4-maverick", help="Model ID")
@click.option("--stream/--no-stream", default=True, help="Stream output")
@click.option("--system", "-s", help="System prompt")
@click.option("--image", "-i", multiple=True, help="Image path or URL (for vision models)")
@click.option("--region", default=None, help="OCI region")
@click.option("--profile", default=None, help="OCI config profile name")
@click.option("--compartment-id", envvar="OCI_COMPARTMENT_ID", help="OCI compartment OCID")
@click.option("--api-key", envvar="OCI_GENAI_API_KEY", help="OCI GenAI API key")
def chat(prompt, model, stream, system, image, region, profile, compartment_id, api_key):
    """Chat with an OCI GenAI model."""
    from oci_genai_service.client import GenAIClient

    client = GenAIClient(
        compartment_id=compartment_id,
        api_key=api_key,
        region=region,
        profile_name=profile,
    )

    if not prompt:
        click.echo("Interactive chat (Ctrl+C to exit)")
        click.echo(f"Model: {model}")
        click.echo()
        try:
            while True:
                prompt = click.prompt("You", prompt_suffix="> ")
                if stream:
                    click.echo("Assistant> ", nl=False)
                    for chunk in client.chat(prompt, model=model, system_prompt=system, stream=True):
                        click.echo(chunk, nl=False)
                    click.echo()
                else:
                    response = client.chat(prompt, model=model, system_prompt=system)
                    click.echo(f"Assistant> {response.text}")
                click.echo()
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
    else:
        images_list = list(image) if image else None
        if stream and not images_list:
            for chunk in client.chat(prompt, model=model, system_prompt=system, stream=True):
                click.echo(chunk, nl=False)
            click.echo()
        else:
            response = client.chat(prompt, model=model, system_prompt=system, images=images_list)
            click.echo(response.text)


@cli.command()
@click.argument("text")
@click.option("--model", "-m", default="cohere.embed-english-v3.0", help="Embedding model ID")
@click.option("--region", default=None, help="OCI region")
@click.option("--profile", default=None, help="OCI config profile name")
@click.option("--compartment-id", envvar="OCI_COMPARTMENT_ID", help="OCI compartment OCID")
def embed(text, model, region, profile, compartment_id):
    """Generate text embeddings."""
    from oci_genai_service.client import GenAIClient

    client = GenAIClient(compartment_id=compartment_id, region=region, profile_name=profile)
    result = client.embed([text], model=model)
    click.echo(f"Model: {model}")
    click.echo(f"Dimensions: {len(result.vectors[0])}")
    click.echo(f"Vector (first 10): {result.vectors[0][:10]}")


@cli.group()
def rag():
    """RAG pipeline commands."""
    pass


@rag.command("ingest")
@click.argument("path")
@click.option("--dsn", required=True, help="Oracle DB DSN (e.g. localhost:1521/FREEPDB1)")
@click.option("--user", default="genai", help="DB username")
@click.option("--password", default="genai", help="DB password")
@click.option("--table", default="documents", help="Vector table name")
@click.option("--model", "-m", default="meta.llama-4-maverick", help="Chat model")
@click.option("--compartment-id", envvar="OCI_COMPARTMENT_ID", help="OCI compartment OCID")
def rag_ingest(path, dsn, user, password, table, model, compartment_id):
    """Ingest documents into the RAG pipeline."""
    from pathlib import Path
    from oci_genai_service.client import GenAIClient
    from oci_genai_service.vectordb.oracle import OracleVectorStore
    from oci_genai_service.rag.pipeline import RAGPipeline

    client = GenAIClient(compartment_id=compartment_id)
    store = OracleVectorStore(dsn=dsn, user=user, password=password, table_name=table)
    pipeline = RAGPipeline(client=client, vector_store=store, chat_model=model)

    p = Path(path)
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.is_file() and f.suffix in {".pdf", ".docx", ".pptx", ".html", ".txt", ".md"}]
    else:
        files = [p]

    total = 0
    for f in files:
        chunks = pipeline.ingest(str(f))
        click.echo(f"Ingested {f.name}: {chunks} chunks")
        total += chunks

    click.echo(f"\nTotal: {total} chunks from {len(files)} files")


@rag.command("query")
@click.argument("question")
@click.option("--dsn", required=True, help="Oracle DB DSN")
@click.option("--user", default="genai", help="DB username")
@click.option("--password", default="genai", help="DB password")
@click.option("--table", default="documents", help="Vector table name")
@click.option("--model", "-m", default="meta.llama-4-maverick", help="Chat model")
@click.option("--compartment-id", envvar="OCI_COMPARTMENT_ID", help="OCI compartment OCID")
def rag_query(question, dsn, user, password, table, model, compartment_id):
    """Query the RAG pipeline."""
    from oci_genai_service.client import GenAIClient
    from oci_genai_service.vectordb.oracle import OracleVectorStore
    from oci_genai_service.rag.pipeline import RAGPipeline

    client = GenAIClient(compartment_id=compartment_id)
    store = OracleVectorStore(dsn=dsn, user=user, password=password, table_name=table)
    pipeline = RAGPipeline(client=client, vector_store=store, chat_model=model)

    result = pipeline.query(question)
    click.echo(result.text)
    if result.sources:
        click.echo(f"\nSources ({len(result.sources)}):")
        for s in result.sources:
            click.echo(f"  - Score: {s['score']:.3f} | {s['chunk'][:80]}...")
