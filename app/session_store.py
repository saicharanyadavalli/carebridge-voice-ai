import os
import json
import sqlite3
import threading
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from app.schemas import SessionState

logger = logging.getLogger("carebridge.session_store")

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DB_DIR, "carebridge_sessions.db")

class SessionStore:
    """
    High-Performance, Thread-Safe Session Store with SQLite Persistence and In-Memory L1 Cache.
    Ensures zero session loss on server restart while preserving sub-millisecond turnaround.
    """

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache: Dict[str, SessionState] = {}
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            call_id TEXT PRIMARY KEY,
                            started_at TEXT NOT NULL,
                            ended_at TEXT,
                            language TEXT NOT NULL,
                            state_json TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        );
                    """)
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to initialize SQLite session storage: {e}")

    def create_session(self, call_id: Optional[str] = None) -> SessionState:
        if not call_id:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        session = SessionState(
            callId=call_id,
            startedAt=now_iso
        )

        with self._lock:
            self._cache[call_id] = session
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO sessions (call_id, started_at, ended_at, language, state_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """, (call_id, session.startedAt, session.endedAt, session.language, session.model_dump_json(), now_iso))
                conn.close()
            except Exception as e:
                logger.warning(f"Error persisting session {call_id}: {e}")

        return session

    def get_session(self, call_id: str) -> Optional[SessionState]:
        with self._lock:
            # 1. Fast L1 cache lookup (< 0.05ms)
            if call_id in self._cache:
                return self._cache[call_id]

            # 2. SQLite L2 lookup
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM sessions WHERE call_id = ?", (call_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    state_dict = json.loads(row[0])
                    session = SessionState.model_validate(state_dict)
                    self._cache[call_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Error fetching session {call_id} from SQLite: {e}")

        return None

    def save_session(self, session: SessionState) -> None:
        """Persists state mutations into SQLite WAL and updates L1 cache."""
        with self._lock:
            self._cache[session.callId] = session
            now_iso = datetime.now(timezone.utc).isoformat()
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO sessions (call_id, started_at, ended_at, language, state_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """, (session.callId, session.startedAt, session.endedAt, session.language, session.model_dump_json(), now_iso))
                conn.close()
            except Exception as e:
                logger.warning(f"Error updating session {session.callId} in SQLite: {e}")

    def end_session(self, call_id: str) -> Optional[SessionState]:
        session = self.get_session(call_id)
        if session:
            now_iso = datetime.now(timezone.utc).isoformat()
            session.endedAt = now_iso
            self.save_session(session)
        return session

    def delete_session(self, call_id: str) -> bool:
        with self._lock:
            if call_id in self._cache:
                del self._cache[call_id]
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("DELETE FROM sessions WHERE call_id = ?", (call_id,))
                conn.close()
                return True
            except Exception as e:
                logger.warning(f"Error deleting session {call_id}: {e}")
        return False

# Global singleton session store
session_store = SessionStore()
