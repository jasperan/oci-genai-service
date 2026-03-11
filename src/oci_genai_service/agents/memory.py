"""Conversation memory backends for agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

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
        if limit:
            return messages[-limit:]
        return list(messages)

    def clear(self, session_id: str) -> None:
        self._sessions[session_id] = []


class OracleMemory(BaseMemory):
    """Oracle-backed conversation memory with vector search for long-term recall."""

    def __init__(self, store: OracleVectorStore, table_name: str = "conversations"):
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
        query = f"""SELECT role, content FROM {self.table_name}
                    WHERE session_id = :session_id ORDER BY created_at"""
        if limit:
            query += f" FETCH LAST {limit} ROWS ONLY"

        with self.store.conn.cursor() as cur:
            cur.execute(query, {"session_id": session_id})
            return [{"role": row[0], "content": row[1]} for row in cur]

    def clear(self, session_id: str) -> None:
        with self.store.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE session_id = :session_id",
                {"session_id": session_id},
            )
        self.store.conn.commit()

    def search(self, query: str, session_id: Optional[str] = None, top_k: int = 5) -> list[dict]:
        """Search conversation history by semantic similarity."""
        results = self.store.search(query, top_k=top_k)
        return [{"role": "context", "content": r.text, "score": r.score} for r in results]
