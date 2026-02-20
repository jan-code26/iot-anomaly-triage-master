from sqlalchemy import text
from backend.database import engine

def main():
    with engine.begin() as conn:
        # Ensure uuid generation support (Neon Postgres usually supports pgcrypto)
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS test_events (
                    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at timestamptz NOT NULL DEFAULT now(),
                    message text NOT NULL
                );
                """
            )
        )

        # Insert one row
        result = conn.execute(
            text(
                """
                INSERT INTO test_events (message)
                VALUES (:message)
                RETURNING id, created_at, message;
                """
            ),
            {"message": "hello from Neon via SQLAlchemy"},
        )
        inserted = result.mappings().one()

        # Read it back
        fetched = conn.execute(
            text(
                """
                SELECT id, created_at, message
                FROM test_events
                WHERE id = :id;
                """
            ),
            {"id": inserted["id"]},
        ).mappings().one()

    print("Inserted:", dict(inserted))
    print("Fetched: ", dict(fetched))

if __name__ == "__main__":
    main()