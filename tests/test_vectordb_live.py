"""Live Oracle integration tests. Skipped unless OGS_TEST_DSN is set.

    OGS_TEST_DSN=127.0.0.1:1527/FREEPDB1 OGS_TEST_USER=... OGS_TEST_PASSWORD=... pytest tests/test_vectordb_live.py
"""

import os
import uuid

import pytest

from oci_genai_service.agents.memory import OracleMemory
from oci_genai_service.vectordb.oracle import OracleVectorStore
from oci_genai_service.vectordb.tables import drop_vector_table

DSN = os.environ.get("OGS_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="OGS_TEST_DSN not set")

DIMS = 4
VECS = {"alpha": [1.0, 0.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0, 0.0], "gamma": [0.0, 0.0, 1.0, 0.0]}


@pytest.fixture
def store():
    name = "OGS_TEST_" + uuid.uuid4().hex[:8].upper()
    s = OracleVectorStore(
        dsn=DSN,
        user=os.environ["OGS_TEST_USER"],
        password=os.environ["OGS_TEST_PASSWORD"],
        table_name=name,
        vector_dims=DIMS,
        embed_fn=lambda texts: [VECS[t] for t in texts],
    )
    try:
        yield s
    finally:
        drop_vector_table(s.conn, name)
        s.close()


def test_add_search_delete(store):
    ids = store.add_texts(["alpha", "beta", "gamma"], metadatas=[{"n": 1}, {"n": 2}, None])
    assert len(ids) == 3 and store.count() == 3

    results = store.search("beta", top_k=2)
    assert results[0].text == "beta"
    assert isinstance(results[0].text, str)
    assert results[0].metadata == {"n": 2}
    assert results[0].score == pytest.approx(1.0)

    assert store.delete(ids[:2]) == 2
    assert store.count() == 1


def test_oracle_memory_roundtrip(store):
    with store.conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE {store.table_config.table_name}_CONV (id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY, "
            "session_id VARCHAR2(255), role VARCHAR2(20), content CLOB, created_at TIMESTAMP DEFAULT SYSTIMESTAMP)"
        )
    conv = store.table_config.table_name + "_CONV"
    try:
        mem = OracleMemory(store, table_name=conv)
        mem.add("s", "user", "hi")
        mem.add("s", "assistant", "hello")
        assert mem.get("s") == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert mem.get("s", limit=1) == [{"role": "assistant", "content": "hello"}]
        mem.clear("s")
        assert mem.get("s") == []
    finally:
        with store.conn.cursor() as cur:
            cur.execute(f"DROP TABLE {conv} PURGE")
