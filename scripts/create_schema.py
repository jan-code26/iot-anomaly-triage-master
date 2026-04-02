"""
Apply the full database schema to your Neon Postgres instance.

Run once (safe to re-run — uses CREATE TABLE IF NOT EXISTS):
    python scripts/create_schema.py

Requires DATABASE_URL to be set in your .env file.
"""
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database import engine
from backend.models import metadata


def main() -> None:
    print("Connecting to Neon Postgres...")
    print(f"  Tables to create: {list(metadata.tables.keys())}\n")

    metadata.create_all(engine)

    print("Done. The following tables now exist in your database:")
    for table_name in metadata.tables:
        print(f"  ✓ {table_name}")


if __name__ == "__main__":
    main()
