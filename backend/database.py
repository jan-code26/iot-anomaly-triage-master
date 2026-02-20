import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file.")

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=0,       # optional but recommended to respect Neon free-tier constraints
    pool_pre_ping=True,
    pool_recycle=300,
)