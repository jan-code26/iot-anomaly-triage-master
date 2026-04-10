"""
Test that the SQLAlchemy engine can connect to Neon Postgres.
"""
from sqlalchemy import text
from backend.database import engine


def test_database_connection():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
