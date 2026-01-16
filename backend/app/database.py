"""
SQLite Persistence Layer for MIA
Stores conversation history, emotion analytics, and user preferences.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading

class MIADatabase:
    """
    SQLite database for MIA persistence.
    Stores:
    - Conversation history
    - Emotion events (for analytics)
    - Session data
    - User preferences
    """
    
    def __init__(self, db_path: str = "mia_data.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.local = threading.local()
        self._init_database()
        print(f"Database initialized: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    def _init_database(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_turns INTEGER DEFAULT 0,
                dominant_emotion TEXT,
                avg_engagement REAL,
                metadata TEXT
            )
        """)
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_text TEXT,
                assistant_response TEXT,
                emotion TEXT,
                confidence REAL,
                modalities TEXT,
                engagement REAL,
                head_pose TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Emotion events (for time-series analytics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                emotion TEXT NOT NULL,
                confidence REAL,
                audio_emotion TEXT,
                text_emotion TEXT,
                video_emotion TEXT,
                engagement REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # User preferences
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analytics aggregates (daily summaries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_analytics (
                date TEXT PRIMARY KEY,
                total_sessions INTEGER,
                total_conversations INTEGER,
                emotion_counts TEXT,
                avg_engagement REAL,
                avg_confidence REAL,
                peak_hour INTEGER
            )
        """)
        
        conn.commit()
    
    # ===================== Session Management =====================
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> str:
        """Create a new session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, metadata)
            VALUES (?, ?)
        """, (session_id, json.dumps(metadata or {})))
        
        conn.commit()
        return session_id
    
    def end_session(self, session_id: str):
        """Mark session as ended and calculate summary stats."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Calculate session stats
        cursor.execute("""
            SELECT emotion, AVG(engagement) as avg_eng
            FROM conversations
            WHERE session_id = ?
            GROUP BY emotion
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """, (session_id,))
        
        row = cursor.fetchone()
        dominant_emotion = row["emotion"] if row else "neutral"
        avg_engagement = row["avg_eng"] if row else 0.5
        
        cursor.execute("""
            SELECT COUNT(*) as total FROM conversations WHERE session_id = ?
        """, (session_id,))
        total_turns = cursor.fetchone()["total"]
        
        cursor.execute("""
            UPDATE sessions
            SET end_time = CURRENT_TIMESTAMP,
                total_turns = ?,
                dominant_emotion = ?,
                avg_engagement = ?
            WHERE session_id = ?
        """, (total_turns, dominant_emotion, avg_engagement, session_id))
        
        conn.commit()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    # ===================== Conversation Storage =====================
    
    def save_conversation(self, 
                         session_id: str,
                         user_text: str,
                         assistant_response: str,
                         emotion: str,
                         confidence: float,
                         modalities: Optional[Dict] = None,
                         engagement: float = 0.5,
                         head_pose: Optional[Dict] = None):
        """Save a conversation turn."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, user_text, assistant_response, emotion, confidence, modalities, engagement, head_pose)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            user_text,
            assistant_response,
            emotion,
            confidence,
            json.dumps(modalities or {}),
            engagement,
            json.dumps(head_pose or {})
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_conversation_history(self, 
                                 session_id: str, 
                                 limit: int = 50) -> List[Dict]:
        """Get conversation history for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    # ===================== Emotion Events =====================
    
    def log_emotion_event(self,
                         session_id: str,
                         emotion: str,
                         confidence: float,
                         audio_emotion: Optional[str] = None,
                         text_emotion: Optional[str] = None,
                         video_emotion: Optional[str] = None,
                         engagement: float = 0.5):
        """Log an emotion detection event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO emotion_events
            (session_id, emotion, confidence, audio_emotion, text_emotion, video_emotion, engagement)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, emotion, confidence, audio_emotion, text_emotion, video_emotion, engagement))
        
        conn.commit()
    
    # ===================== Analytics =====================
    
    def get_emotion_distribution(self, 
                                 days: int = 7,
                                 session_id: Optional[str] = None) -> Dict[str, int]:
        """Get emotion distribution over time."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT emotion, COUNT(*) as count
                FROM emotion_events
                WHERE session_id = ?
                GROUP BY emotion
            """, (session_id,))
        else:
            cursor.execute("""
                SELECT emotion, COUNT(*) as count
                FROM emotion_events
                WHERE timestamp > datetime('now', ?)
                GROUP BY emotion
            """, (f'-{days} days',))
        
        rows = cursor.fetchall()
        return {row["emotion"]: row["count"] for row in rows}
    
    def get_emotion_timeline(self,
                            session_id: Optional[str] = None,
                            hours: int = 24) -> List[Dict]:
        """Get emotion events over time for timeline visualization."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT timestamp, emotion, confidence, engagement
                FROM emotion_events
                WHERE session_id = ?
                ORDER BY timestamp
            """, (session_id,))
        else:
            cursor.execute("""
                SELECT timestamp, emotion, confidence, engagement
                FROM emotion_events
                WHERE timestamp > datetime('now', ?)
                ORDER BY timestamp
            """, (f'-{hours} hours',))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_engagement_stats(self, days: int = 7) -> Dict:
        """Get engagement statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(engagement) as avg_engagement,
                MIN(engagement) as min_engagement,
                MAX(engagement) as max_engagement,
                COUNT(*) as total_events
            FROM emotion_events
            WHERE timestamp > datetime('now', ?)
        """, (f'-{days} days',))
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def get_session_summary(self, days: int = 30) -> List[Dict]:
        """Get summary of recent sessions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                session_id,
                start_time,
                end_time,
                total_turns,
                dominant_emotion,
                avg_engagement
            FROM sessions
            WHERE start_time > datetime('now', ?)
            ORDER BY start_time DESC
        """, (f'-{days} days',))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_daily_summary(self, date: str = None) -> Dict:
        """Get or generate daily summary."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if we have cached summary
        cursor.execute("SELECT * FROM daily_analytics WHERE date = ?", (date,))
        cached = cursor.fetchone()
        
        if cached:
            result = dict(cached)
            result["emotion_counts"] = json.loads(result["emotion_counts"])
            return result
        
        # Generate summary
        cursor.execute("""
            SELECT COUNT(DISTINCT session_id) as sessions
            FROM conversations
            WHERE DATE(timestamp) = ?
        """, (date,))
        total_sessions = cursor.fetchone()["sessions"]
        
        cursor.execute("""
            SELECT COUNT(*) as convos
            FROM conversations
            WHERE DATE(timestamp) = ?
        """, (date,))
        total_conversations = cursor.fetchone()["convos"]
        
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM emotion_events
            WHERE DATE(timestamp) = ?
            GROUP BY emotion
        """, (date,))
        emotion_counts = {row["emotion"]: row["count"] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT AVG(engagement) as eng, AVG(confidence) as conf
            FROM emotion_events
            WHERE DATE(timestamp) = ?
        """, (date,))
        avgs = cursor.fetchone()
        
        summary = {
            "date": date,
            "total_sessions": total_sessions,
            "total_conversations": total_conversations,
            "emotion_counts": emotion_counts,
            "avg_engagement": avgs["eng"] or 0,
            "avg_confidence": avgs["conf"] or 0
        }
        
        return summary
    
    # ===================== Preferences =====================
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO preferences (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, json.dumps(value)))
        
        conn.commit()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM preferences WHERE key = ?", (key,))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row["value"])
        return default
    
    def get_all_preferences(self) -> Dict:
        """Get all preferences."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT key, value FROM preferences")
        rows = cursor.fetchall()
        
        return {row["key"]: json.loads(row["value"]) for row in rows}
    
    # ===================== Cleanup =====================
    
    def cleanup_old_data(self, days: int = 90):
        """Remove data older than specified days."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM emotion_events
            WHERE timestamp < datetime('now', ?)
        """, (f'-{days} days',))
        
        cursor.execute("""
            DELETE FROM conversations
            WHERE timestamp < datetime('now', ?)
        """, (f'-{days} days',))
        
        cursor.execute("""
            DELETE FROM sessions
            WHERE start_time < datetime('now', ?)
        """, (f'-{days} days',))
        
        conn.commit()
        conn.execute("VACUUM")  # Reclaim space
    
    def close(self):
        """Close database connection."""
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None
