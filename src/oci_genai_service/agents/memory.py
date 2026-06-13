"""Conversation memory backends for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from oci_genai_service.vectordb.oracle import OracleVectorStore


class BaseMemory(ABC):
    """Abstract base class for conversation memory."""

    @abstractmethod
    def add(self, session_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    def get(self, session_id: str, limit: Optional[int] = None) -> list[dict]: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...


class InMemoryMemory(BaseMemory):
    """Simple in-memory conversation history."""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def add(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append({"role": role, "content": content})

    def get(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        messages = self._sessions[session_id]
        if limit is not None:
            return messages[-limit:]
        return list(messages)

    def clear(self, session_id: str) -> None:
        self._sessions[session_id] = []


class OracleMemory(BaseMemory):
    """Oracle-backed conversation memory with vector search for long-term recall."""

    def __init__(self, store: "OracleVectorStore", table_name: str = "conversations"):
        from oci_genai_service.vectordb.tables import validate_identifier
        validate_identifier(table_name)
        self.store = store
        self.table_name = table_name

    def add(self, session_id: str, role: str, content: str) -> None:
        with self.store.conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {self.table_name} (session_id, role, content)
                VALUES (:session_id, :role, :content)""",
                {"session_id": session_id, "role": role, "content": content},
            )
        self.store.conn.commit()

    def get(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        params: dict = {"session_id": session_id}
        if limit is not None:
            # Fetch the most recent `limit` rows, then reverse back to chronological order.
            query = f"""SELECT role, content FROM {self.table_name}
                        WHERE session_id = :session_id
                        ORDER BY created_at DESC
                        FETCH FIRST :limit ROWS ONLY"""
            params["limit"] = limit
        else:
            query = f"""SELECT role, content FROM {self.table_name}
                        WHERE session_id = :session_id
                        ORDER BY created_at"""
        with self.store.conn.cursor() as cur:
            cur.execute(query, params)
            results = [{"role": row[0], "content": row[1]} for row in cur]
        return list(reversed(results)) if limit is not None else results

    def clear(self, session_id: str) -> None:
        with self.store.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE session_id = :session_id",
                {"session_id": session_id},
            )
        self.store.conn.commit()

