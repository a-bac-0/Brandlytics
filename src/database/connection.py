import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, Optional
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database connection and session management
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _get_database_url(self) -> str:
        """Construct database URL from config or environment variables"""
        db_config = self.config.get('database', {})
        
        # Use environment variables if available, otherwise use config
        host = os.getenv('DB_HOST', db_config.get('host', 'localhost'))
        port = os.getenv('DB_PORT', db_config.get('port', 5432))
        database = os.getenv('DB_NAME', db_config.get('name', 'brand_detection'))
        username = os.getenv('DB_USER', db_config.get('user', 'postgres'))
        password = os.getenv('DB_PASSWORD', db_config.get('password', ''))
        
        if db_config.get('type') == 'sqlite':
            return f"sqlite:///{database}.db"
        else:
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            database_url = self._get_database_url()
            logger.info(f"Connecting to database...")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            self._create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
