# Unified Crime Detection Database and Alert System

A comprehensive SQLite database schema and Telegram alert system designed for the Real-Time Crime Detection and Prevention System. This unified backend handles events from all four detection modules: ANPR, Face Recognition, Violence Detection, and Weapon Detection.

## Features

### 🗄️ **Unified Database Schema**
- **Comprehensive Event Logging**: All detection events stored in normalized SQLite database
- **Module-Specific Tables**: Dedicated tables for ANPR, Face, Violence, and Weapon detection data
- **Advanced Analytics**: Built-in support for dashboard analytics and reporting
- **Smart Search**: Full-text search and advanced filtering capabilities
- **Performance Optimized**: Indexed tables with query optimization

### 🚨 **Intelligent Alert System**
- **Multi-Channel Alerts**: Telegram, Email, SMS, and Webhook support
- **Smart Rate Limiting**: Prevents alert spam with intelligent cooldowns
- **Priority-Based Queue**: Critical alerts processed first
- **Duplicate Detection**: Avoids sending duplicate alerts
- **Template Engine**: Customizable alert messages for each detection type

### 📊 **Analytics & Reporting**
- **Real-Time Metrics**: Live system performance and activity monitoring
- **Dashboard Analytics**: Comprehensive metrics for frontend dashboards
- **Trend Analysis**: Historical data analysis and trend detection
- **Custom Reports**: Security summaries, performance analysis, and compliance reports
- **Data Export**: CSV and JSON export capabilities

### ⚙️ **System Management**
- **Health Monitoring**: Automatic system health checks and alerts
- **Configuration Management**: Environment-based configuration with validation
- **Connection Pooling**: Optimized database connections with automatic failover
- **Backup Management**: Automated database backups with retention policies
- **Migration Support**: Easy database schema updates and data migration

## Quick Start

### 1. Installation

```bash
# Navigate to the backend directory
cd backend/

# Set environment variables
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
export DATABASE_PATH="crime_detection.db"
```

### 2. Initialize the System

```python
from database import initialize_database
from services.database_service import get_database_service

# Initialize database schema
initialize_database("crime_detection.db")

# Get database service
db_service = get_database_service("crime_detection.db")
```

### 3. Log Detection Events

```python
# Example: Log ANPR detection
detection_data = {
    'plate_number': 'ABC123',
    'confidence': 0.95,
    'location': 'Main_Gate',
    'is_red_listed': False,
    'bounding_box': {'x1': 100, 'y1': 50, 'x2': 200, 'y2': 100}
}

event_id = db_service.log_detection_event('ANPR', detection_data)
```

### 4. Run the Demo

```bash
python integration_example.py
```

## Database Schema

### Core Tables

- **`events`** - Master events table for all detection types
- **`detections`** - Detection details with bounding boxes
- **`alerts`** - Alert tracking and delivery status
- **`red_listed_items`** - Red-listed vehicles, persons, and objects
- **`system_logs`** - Comprehensive system logging

### Module-Specific Tables

- **`anpr_events`** - License plate recognition data
- **`face_events`** - Face recognition and categorization
- **`violence_events`** - Violence detection with intensity metrics
- **`weapon_events`** - Weapon detection with threat levels

### Analytics Tables

- **`detection_statistics`** - Aggregated statistics for dashboard
- **`search_index`** - Full-text search optimization

## Alert Templates

The system includes pre-built alert templates for each detection type:

### ANPR Alerts
- **Red Alert**: Critical alert for red-listed vehicles
- **Detection**: Standard vehicle detection notification

### Face Recognition Alerts  
- **Criminal Alert**: Critical alert for known criminals
- **Restricted Alert**: Security alert for restricted persons
- **Unknown Detection**: Notification for unidentified persons

### Violence Detection Alerts
- **Violence Alert**: Emergency alert for violent incidents

### Weapon Detection Alerts
- **Critical Alert**: Maximum threat level alerts
- **High Alert**: High priority weapon detections
- **Detection**: Standard weapon detection notifications

### System Alerts
- **Startup/Shutdown**: System status notifications
- **Error Alerts**: Technical issue notifications
- **Daily Reports**: Comprehensive activity summaries

## Configuration

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=sqlite:///crime_detection.db
DATABASE_PATH=crime_detection.db
DB_POOL_SIZE=10
DB_BACKUP_INTERVAL_HOURS=24

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Alert Configuration
MAX_ALERTS_PER_MINUTE=10
ALERT_COOLDOWN_SECONDS=60
ENABLE_DUPLICATE_DETECTION=true

# Performance Settings
DB_CACHE_SIZE=-64000
SQLITE_JOURNAL_MODE=WAL
ANALYTICS_CACHE_TTL=300
```

### Database Configuration File

```python
from database.config import DatabaseConfig

config = DatabaseConfig(
    database_path="crime_detection.db",
    pool_size=10,
    enable_backups=True,
    backup_interval_hours=24,
    sqlite_journal_mode="WAL"
)
```

## API Integration

### Enhanced Service Wrappers

The system provides enhanced versions of existing service wrappers:

```python
from services.enhanced_anpr_service import EnhancedANPRService

# Initialize with database integration
anpr_service = EnhancedANPRService(enable_database=True)

# Detections are automatically logged and alerts sent
result = await anpr_service.detect_plates(image, min_confidence=0.7)
```

### Analytics API

```python
from services.analytics_service import AnalyticsService, TimeRange

analytics = AnalyticsService(db_operations)

# Get dashboard overview
overview = analytics.get_dashboard_overview(TimeRange.DAY)

# Search events
search_results = analytics.search_events(search_query)

# Generate reports
report = analytics.generate_report('security_summary', TimeRange.WEEK)
```

## Alert System Usage

### Manual Alerts

```python
from services.alert_service import AlertPriority

# Send system alert
success = alert_manager.send_system_alert(
    "System maintenance scheduled for tonight",
    AlertPriority.MEDIUM
)

# Send detection alert
success = alert_manager.send_detection_alert(
    event, AlertType.TELEGRAM, AlertPriority.HIGH
)
```

### Custom Templates

```python
from services.alert_templates import get_template_engine, TemplateType

template_engine = get_template_engine()

# Render custom alert
rendered = template_engine.render_template(
    TemplateType.ANPR_RED_ALERT,
    {
        'plate_number': 'XYZ789',
        'location': 'Main Gate',
        'alert_reason': 'Stolen Vehicle'
    }
)
```

## Red-Listed Items Management

```python
# Add red-listed vehicle
success = db_service.add_red_listed_item(
    'VEHICLE', 'ABC123', 'Suspicious Activity', 'HIGH'
)

# Check if item is red-listed
is_red_listed, reason = db_service.is_red_listed('VEHICLE', 'ABC123')

# Remove from red-list
success = db_service.remove_red_listed_item('VEHICLE', 'ABC123')

# Get all red-listed items
items = db_service.get_red_listed_items('VEHICLE')
```

## System Health Monitoring

```python
# Get system health
health = db_service.get_system_health()

# Test all connections
results = db_service.test_connections()

# Get performance metrics
metrics = analytics.get_real_time_metrics()
```

## Data Export and Backup

```python
# Export search results to CSV
export_data = analytics.export_data(search_query, format_type='csv')

# Manual backup
backup_stats = db_service.cleanup_old_data(retention_days=90)

# Get connection info
conn_info = connection_manager.get_connection_info()
```

## Best Practices

### 1. **Error Handling**
Always wrap database operations in try-catch blocks and check return values.

### 2. **Resource Management**
Use context managers or ensure proper cleanup of database sessions.

### 3. **Configuration**
Use environment variables for sensitive configuration like API tokens.

### 4. **Monitoring**
Regularly check system health and alert service statistics.

### 5. **Backup Strategy**
Enable automatic backups and test restoration procedures.

### 6. **Alert Tuning**
Adjust alert thresholds and rate limiting based on your requirements.

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check database file permissions
   - Verify SQLite installation
   - Check disk space

2. **Alert Delivery Issues**
   - Verify Telegram bot token and chat ID
   - Check network connectivity
   - Review rate limiting settings

3. **Performance Issues**
   - Check database indexes
   - Review query patterns
   - Monitor memory usage

### Logging

The system provides comprehensive logging. Check logs for detailed error information:

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Support

For issues or questions:

1. Check the logs for error details
2. Review the integration example
3. Verify configuration settings
4. Test individual components

## License

This unified system is part of the Real-Time Crime Detection and Prevention System project.