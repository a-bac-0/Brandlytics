#!/usr/bin/env python3
"""
Database setup script for Brand Detection project
"""
import os
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.connection import DatabaseManager
from src.database.models import Base

def main():
    """Initialize database and create tables"""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info("Setting up database...")
        
        # Initialize database
        db_manager = DatabaseManager(config)
        
        logger.info("Database setup completed successfully!")
        
        # Test connection
        if db_manager.health_check():
            logger.info("✅ Database health check passed")
        else:
            logger.error("❌ Database health check failed")
            return 1
        
        return 0
        
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())