"""
Database Configuration and Connection Management
Provides centralized database configuration, connection pooling, and environment-based settings
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool, QueuePool
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Connection settings
    database_url: str = "sqlite:///crime_detection.db"
    database_path: str = "crime_detection.db"
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    
    # SQLite specific settings
    sqlite_timeout: int = 20
    sqlite_check_same_thread: bool = False
    sqlite_foreign_keys: bool = True
    sqlite_journal_mode: str = "WAL"  # Write-Ahead Logging
    sqlite_synchronous: str = "NORMAL"
    sqlite_cache_size: int = -64000  # 64MB cache
    
    # Performance settings
    echo: bool = False
    echo_pool: bool = False
    connect_args: Dict[str, Any] = field(default_factory=dict)
    
    # Backup settings
    enable_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_directory: str = "backups"
    
    # Maintenance settings
    enable_auto_vacuum: bool = True
    vacuum_interval_hours: int = 168  # Weekly
    analyze_interval_hours: int = 24
    
    # Health check settings
    health_check_interval_seconds: int = 300  # 5 minutes
    max_failed_health_checks: int = 3
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.connect_args:
            self.connect_args = self._get_default_connect_args()
    
    def _get_default_connect_args(self) -> Dict[str, Any]:
        """Get default connection arguments based on database type."""
        if self.database_url.startswith('sqlite'):
            return {
                'timeout': self.sqlite_timeout,
                'check_same_thread': self.sqlite_check_same_thread,
            }
        return {}
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Load from environment
        config.database_url = os.getenv('DATABASE_URL', config.database_url)
        config.database_path = os.getenv('DATABASE_PATH', config.database_path)
        
        # Pool settings
        config.pool_size = int(os.getenv('DB_POOL_SIZE', config.pool_size))
        config.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', config.max_overflow))
        config.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', config.pool_timeout))
        config.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', config.pool_recycle))
        
        # SQLite settings
        config.sqlite_timeout = int(os.getenv('SQLITE_TIMEOUT', config.sqlite_timeout))
        config.sqlite_journal_mode = os.getenv('SQLITE_JOURNAL_MODE', config.sqlite_journal_mode)
        config.sqlite_synchronous = os.getenv('SQLITE_SYNCHRONOUS', config.sqlite_synchronous)
        config.sqlite_cache_size = int(os.getenv('SQLITE_CACHE_SIZE', config.sqlite_cache_size))
        
        # Boolean settings
        config.echo = os.getenv('DB_ECHO', 'false').lower() == 'true'
        config.echo_pool = os.getenv('DB_ECHO_POOL', 'false').lower() == 'true'
        config.sqlite_foreign_keys = os.getenv('SQLITE_FOREIGN_KEYS', 'true').lower() == 'true'
        
        # Backup settings
        config.enable_backups = os.getenv('DB_ENABLE_BACKUPS', 'true').lower() == 'true'
        config.backup_interval_hours = int(os.getenv('DB_BACKUP_INTERVAL_HOURS', config.backup_interval_hours))
        config.backup_retention_days = int(os.getenv('DB_BACKUP_RETENTION_DAYS', config.backup_retention_days))
        config.backup_directory = os.getenv('DB_BACKUP_DIRECTORY', config.backup_directory)
        
        return config
    
    @classmethod
    def from_file(cls, config_file: str) -> 'DatabaseConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            config = cls()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from file {config_file}: {e}")
            return cls()
    
    def save_to_file(self, config_file: str) -> bool:
        """Save configuration to JSON file."""
        try:
            config_data = {
                'database_url': self.database_url,
                'database_path': self.database_path,
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
                'sqlite_timeout': self.sqlite_timeout,
                'sqlite_journal_mode': self.sqlite_journal_mode,
                'sqlite_synchronous': self.sqlite_synchronous,
                'sqlite_cache_size': self.sqlite_cache_size,
                'echo': self.echo,
                'enable_backups': self.enable_backups,
                'backup_interval_hours': self.backup_interval_hours,
                'backup_retention_days': self.backup_retention_days,
                'backup_directory': self.backup_directory
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to file {config_file}: {e}")
            return False
    
    def validate(self) -> List[str]:
        """Validate configuration settings."""
        errors = []
        
        # Validate pool settings
        if self.pool_size <= 0:
            errors.append("pool_size must be greater than 0")
        if self.max_overflow < 0:
            errors.append("max_overflow must be non-negative")
        if self.pool_timeout <= 0:
            errors.append("pool_timeout must be greater than 0")
        
        # Validate SQLite settings
        if self.sqlite_timeout <= 0:
            errors.append("sqlite_timeout must be greater than 0")
        if self.sqlite_journal_mode not in ['DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF']:
            errors.append(f"Invalid sqlite_journal_mode: {self.sqlite_journal_mode}")
        if self.sqlite_synchronous not in ['OFF', 'NORMAL', 'FULL', 'EXTRA']:
            errors.append(f"Invalid sqlite_synchronous: {self.sqlite_synchronous}")
        
        # Validate backup settings
        if self.enable_backups:
            if self.backup_interval_hours <= 0:
                errors.append("backup_interval_hours must be greater than 0")
            if self.backup_retention_days <= 0:
                errors.append("backup_retention_days must be greater than 0")
        
        return errors

class DatabaseConnectionManager:
    """
    Advanced database connection manager with pooling, health monitoring, and maintenance.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory = None
        self.scoped_session = None
        
        # Health monitoring
        self.health_check_thread = None
        self.health_check_running = False
        self.failed_health_checks = 0
        self.last_health_check = None
        self.is_healthy = True
        
        # Backup management
        self.backup_thread = None
        self.backup_running = False
        self.last_backup = None
        
        # Maintenance
        self.maintenance_thread = None
        self.maintenance_running = False
        self.last_vacuum = None
        self.last_analyze = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        self._initialize_connection()
        self._start_background_tasks()
    
    def _initialize_connection(self):
        """Initialize database connection and engine."""
        try:
            # Create engine based on database type
            if self.config.database_url.startswith('sqlite'):
                self.engine = self._create_sqlite_engine()
            else:
                self.engine = self._create_generic_engine()
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            self.scoped_session = scoped_session(self.session_factory)
            
            # Setup event listeners
            self._setup_event_listeners()
            
            logger.info(f"Database connection initialized: {self.config.database_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    def _create_sqlite_engine(self) -> Engine:
        """Create SQLite engine with optimized settings."""
        # Ensure database directory exists
        db_path = Path(self.config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        connect_args = {
            'timeout': self.config.sqlite_timeout,
            'check_same_thread': self.config.sqlite_check_same_thread,
        }
        
        engine = create_engine(
            self.config.database_url,
            echo=self.config.echo,
            echo_pool=self.config.echo_pool,
            poolclass=StaticPool,
            connect_args=connect_args,
            pool_pre_ping=self.config.pool_pre_ping
        )
        
        return engine
    
    def _create_generic_engine(self) -> Engine:
        """Create generic database engine with connection pooling."""
        engine = create_engine(
            self.config.database_url,
            echo=self.config.echo,
            echo_pool=self.config.echo_pool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            connect_args=self.config.connect_args
        )
        
        return engine
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for optimization."""
        if self.config.database_url.startswith('sqlite'):
            # SQLite-specific optimizations
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                
                # Enable foreign keys
                if self.config.sqlite_foreign_keys:
                    cursor.execute("PRAGMA foreign_keys=ON")
                
                # Set journal mode
                cursor.execute(f"PRAGMA journal_mode={self.config.sqlite_journal_mode}")
                
                # Set synchronous mode
                cursor.execute(f"PRAGMA synchronous={self.config.sqlite_synchronous}")
                
                # Set cache size
                cursor.execute(f"PRAGMA cache_size={self.config.sqlite_cache_size}")
                
                # Enable memory mapping
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                
                # Optimize for better performance
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA optimize")
                
                cursor.close()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.config.health_check_interval_seconds > 0:
            self.health_check_running = True
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop, 
                daemon=True
            )
            self.health_check_thread.start()
        
        if self.config.enable_backups:
            self.backup_running = True
            self.backup_thread = threading.Thread(
                target=self._backup_loop,
                daemon=True
            )
            self.backup_thread.start()
        
        if self.config.enable_auto_vacuum:
            self.maintenance_running = True
            self.maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                daemon=True
            )
            self.maintenance_thread.start()
    
    def _health_check_loop(self):
        """Background health check loop."""
        while self.health_check_running:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def _perform_health_check(self):
        """Perform database health check."""
        try:
            with self._lock:
                session = self.get_session()
                try:
                    # Simple query to test connection
                    session.execute(text("SELECT 1"))
                    session.commit()
                    
                    # Health check passed
                    self.failed_health_checks = 0
                    self.is_healthy = True
                    self.last_health_check = time.time()
                    
                except Exception as e:
                    logger.warning(f"Database health check failed: {e}")
                    self.failed_health_checks += 1
                    
                    if self.failed_health_checks >= self.config.max_failed_health_checks:
                        self.is_healthy = False
                        logger.error("Database marked as unhealthy after multiple failed checks")
                
                finally:
                    self.close_session(session)
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.failed_health_checks += 1
    
    def _backup_loop(self):
        """Background backup loop."""
        while self.backup_running:
            try:
                if self._should_perform_backup():
                    self._perform_backup()
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
                time.sleep(3600)
    
    def _should_perform_backup(self) -> bool:
        """Check if backup should be performed."""
        if not self.config.enable_backups:
            return False
        
        if self.last_backup is None:
            return True
        
        elapsed_hours = (time.time() - self.last_backup) / 3600
        return elapsed_hours >= self.config.backup_interval_hours
    
    def _perform_backup(self):
        """Perform database backup."""
        try:
            if not self.config.database_url.startswith('sqlite'):
                logger.info("Backup only supported for SQLite databases")
                return
            
            # Create backup directory
            backup_dir = Path(self.config.backup_directory)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"crime_detection_backup_{timestamp}.db"
            backup_path = backup_dir / backup_filename
            
            # Perform backup using SQLite VACUUM INTO
            with self._lock:
                session = self.get_session()
                try:
                    session.execute(text(f"VACUUM INTO '{backup_path}'"))
                    session.commit()
                    
                    self.last_backup = time.time()
                    logger.info(f"Database backup created: {backup_path}")
                    
                    # Cleanup old backups
                    self._cleanup_old_backups()
                    
                except Exception as e:
                    logger.error(f"Backup failed: {e}")
                finally:
                    self.close_session(session)
                    
        except Exception as e:
            logger.error(f"Backup process error: {e}")
    
    def _cleanup_old_backups(self):
        """Remove old backup files."""
        try:
            backup_dir = Path(self.config.backup_directory)
            if not backup_dir.exists():
                return
            
            cutoff_time = time.time() - (self.config.backup_retention_days * 24 * 3600)
            
            for backup_file in backup_dir.glob("crime_detection_backup_*.db"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.info(f"Removed old backup: {backup_file}")
                    
        except Exception as e:
            logger.error(f"Backup cleanup error: {e}")
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.maintenance_running:
            try:
                # Check for vacuum
                if self._should_perform_vacuum():
                    self._perform_vacuum()
                
                # Check for analyze
                if self._should_perform_analyze():
                    self._perform_analyze()
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                time.sleep(3600)
    
    def _should_perform_vacuum(self) -> bool:
        """Check if vacuum should be performed."""
        if self.last_vacuum is None:
            return True
        
        elapsed_hours = (time.time() - self.last_vacuum) / 3600
        return elapsed_hours >= self.config.vacuum_interval_hours
    
    def _should_perform_analyze(self) -> bool:
        """Check if analyze should be performed."""
        if self.last_analyze is None:
            return True
        
        elapsed_hours = (time.time() - self.last_analyze) / 3600
        return elapsed_hours >= self.config.analyze_interval_hours
    
    def _perform_vacuum(self):
        """Perform database vacuum."""
        try:
            if not self.config.database_url.startswith('sqlite'):
                return
            
            with self._lock:
                session = self.get_session()
                try:
                    logger.info("Starting database vacuum...")
                    session.execute(text("VACUUM"))
                    session.commit()
                    
                    self.last_vacuum = time.time()
                    logger.info("Database vacuum completed")
                    
                except Exception as e:
                    logger.error(f"Vacuum failed: {e}")
                finally:
                    self.close_session(session)
                    
        except Exception as e:
            logger.error(f"Vacuum process error: {e}")
    
    def _perform_analyze(self):
        """Perform database analyze."""
        try:
            if not self.config.database_url.startswith('sqlite'):
                return
            
            with self._lock:
                session = self.get_session()
                try:
                    logger.info("Starting database analyze...")
                    session.execute(text("ANALYZE"))
                    session.commit()
                    
                    self.last_analyze = time.time()
                    logger.info("Database analyze completed")
                    
                except Exception as e:
                    logger.error(f"Analyze failed: {e}")
                finally:
                    self.close_session(session)
                    
        except Exception as e:
            logger.error(f"Analyze process error: {e}")
    
    # Public interface methods
    def get_session(self):
        """Get a database session."""
        if not self.session_factory:
            raise RuntimeError("Database connection not initialized")
        return self.session_factory()
    
    def get_scoped_session(self):
        """Get a scoped session (thread-local)."""
        if not self.scoped_session:
            raise RuntimeError("Database connection not initialized")
        return self.scoped_session()
    
    def close_session(self, session):
        """Close a database session."""
        if session:
            session.close()
    
    def remove_scoped_session(self):
        """Remove the current scoped session."""
        if self.scoped_session:
            self.scoped_session.remove()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information and statistics."""
        info = {
            'database_url': self.config.database_url,
            'database_path': self.config.database_path,
            'is_healthy': self.is_healthy,
            'failed_health_checks': self.failed_health_checks,
            'last_health_check': self.last_health_check,
            'last_backup': self.last_backup,
            'last_vacuum': self.last_vacuum,
            'last_analyze': self.last_analyze
        }
        
        if self.engine:
            info['pool_size'] = getattr(self.engine.pool, 'size', lambda: 'N/A')()
            info['checked_out_connections'] = getattr(self.engine.pool, 'checkedout', lambda: 'N/A')()
            info['pool_status'] = str(self.engine.pool.status())
        
        return info
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            session = self.get_session()
            try:
                session.execute(text("SELECT 1"))
                return True
            finally:
                self.close_session(session)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the connection manager."""
        logger.info("Shutting down database connection manager...")
        
        # Stop background tasks
        self.health_check_running = False
        self.backup_running = False
        self.maintenance_running = False
        
        # Wait for threads to finish
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)
        
        if self.backup_thread and self.backup_thread.is_alive():
            self.backup_thread.join(timeout=5)
        
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5)
        
        # Remove scoped sessions
        if self.scoped_session:
            self.scoped_session.remove()
        
        # Dispose engine
        if self.engine:
            self.engine.dispose()
        
        logger.info("Database connection manager shutdown complete")

# Global connection manager
_connection_manager: Optional[DatabaseConnectionManager] = None

def get_connection_manager(config: Optional[DatabaseConfig] = None) -> DatabaseConnectionManager:
    """Get the global database connection manager."""
    global _connection_manager
    
    if _connection_manager is None:
        if config is None:
            config = DatabaseConfig.from_env()
        _connection_manager = DatabaseConnectionManager(config)
    
    return _connection_manager

def initialize_database_connection(config: Optional[DatabaseConfig] = None) -> DatabaseConnectionManager:
    """Initialize the global database connection manager."""
    global _connection_manager
    
    if config is None:
        config = DatabaseConfig.from_env()
    
    _connection_manager = DatabaseConnectionManager(config)
    return _connection_manager

def shutdown_database_connection():
    """Shutdown the global database connection manager."""
    global _connection_manager
    
    if _connection_manager:
        _connection_manager.shutdown()
        _connection_manager = None