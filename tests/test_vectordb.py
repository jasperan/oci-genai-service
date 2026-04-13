"""Tests for Oracle Vector Search module."""

from unittest.mock import patch, MagicMock
from oci_genai_service.vectordb.oracle import OracleVectorStore
from oci_genai_service.vectordb.tables import VectorTableConfig


class TestVectorTableConfig:
    def test_default_config(self):
        config = VectorTableConfig(table_name="documents")
        assert config.vector_dims == 1024
        assert config.distance_metric == "COSINE"
        assert config.id_column == "id"
        assert config.text_column == "text"
        assert config.vector_column == "embedding"
        assert config.metadata_column == "metadata"

    def test_custom_dims(self):
        config = VectorTableConfig(table_name="docs", vector_dims=384)
        assert config.vector_dims == 384


class TestOracleVectorStore:
    @patch("oci_genai_service.vectordb.oracle.create_vector_table")
    @patch("oci_genai_service.vectordb.oracle.oracledb")
    def test_init_creates_connection(self, mock_oracledb, mock_create_table):
        mock_conn = MagicMock()
        mock_oracledb.connect.return_value = mock_conn

        OracleVectorStore(
            dsn="localhost:1521/FREEPDB1",
            user="genai",
            password="genai",
            table_name="documents",
        )
        mock_oracledb.connect.assert_called_once()

    @patch("oci_genai_service.vectordb.oracle.create_vector_table")
    @patch("oci_genai_service.vectordb.oracle.oracledb")
    def test_add_texts_inserts_rows(self, mock_oracledb, mock_create_table):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_oracledb.connect.return_value = mock_conn

        store = OracleVectorStore(
            dsn="localhost:1521/FREEPDB1",
            user="genai", password="genai",
            table_name="documents",
        )
        store._embed_fn = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        store.add_texts(["hello", "world"])
        assert mock_cursor.execute.called
