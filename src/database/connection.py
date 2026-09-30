"""Database connection, engine configuration, and session management for ChequeSense."""

from __future__ import annotations

import logging
import os
import urllib.parse
from contextlib import contextmanager
from typing import Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

logger = logging.getLogger("chequesense.database.connection")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def build_database_url() -> str:
    """Builds database URL from environment variables, never hardcoding credentials."""
    # 1. Direct DATABASE_URL environment override
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    # 2. Reconstruct from discrete parameters
    user = os.getenv("POSTGRES_USER", "postgres")
    raw_password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "chequesense")

    if raw_password:
        password_encoded = urllib.parse.quote_plus(raw_password)
        auth = f"{user}:{password_encoded}@"
    elif user:
        auth = f"{user}@"
    else:
        auth = ""

    return f"postgresql+psycopg2://{auth}{host}:{port}/{db_name}"


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Creates a SQLAlchemy engine with connection pooling and ping validation."""
    url = db_url or build_database_url()

    # SQLite config vs PostgreSQL config
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    else:
        engine = create_engine(
            url,
            pool_pre_ping=True,      # Validates connection liveness before checkout
            pool_size=10,            # Base connection pool size
            max_overflow=20,         # Maximum temporary overflow connections
            pool_recycle=3600,       # Recycles idle connections hourly
            echo=False,
        )

    logger.debug("Configured database engine for: %s", url.split("@")[-1])
    return engine


# Global engine and session factory
_default_engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_default_engine)


def get_db(custom_engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Yields a database session with automatic commit/rollback and cleanup."""
    factory = sessionmaker(autocommit=False, autoflush=False, bind=custom_engine) if custom_engine else SessionLocal
    session: Session = factory()
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error("Database session rolled back due to error: %s", e)
        raise
    finally:
        session.close()


@contextmanager
def db_session_scope(custom_engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager for explicit transaction scopes."""
    factory = sessionmaker(autocommit=False, autoflush=False, bind=custom_engine) if custom_engine else SessionLocal
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_db_connection(engine: Optional[Engine] = None) -> bool:
    """Verifies that the database server is reachable."""
    eng = engine or _default_engine
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("Database connection check failed: %s", e)
        return False


def init_db(engine: Optional[Engine] = None) -> None:
    """Initializes database schema and creates all tables."""
    eng = engine or _default_engine
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=eng)
    logger.info("Database tables initialized successfully.")
