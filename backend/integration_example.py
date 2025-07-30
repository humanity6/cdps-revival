"""
Integration Example for Unified Crime Detection Database and Alert System
Demonstrates how to use the complete system with all modules
"""

import sys
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add backend to path
sys.path.append(os.path.dirname(__file__))

# Import unified system components
from database import (
    initialize_database, DatabaseConfig, initialize_database_connection,
    get_connection_manager, DatabaseOperations
)
from services.database_service import DatabaseService, get_database_service
from services.analytics_service import AnalyticsService, TimeRange, SearchQuery, SearchFilter
from services.alert_service import AlertConfiguration, UnifiedAlertManager, AlertPriority, AlertType
from services.alert_templates import get_template_engine, TemplateType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrimeDetectionSystemExample:
    """
    Example implementation showing how to integrate all components
    of the unified crime detection database and alert system.
    """
    
    def __init__(self):
        self.db_service = None
        self.analytics_service = None
        self.alert_manager = None
        self.template_engine = None
        
    async def initialize_system(self):
        """Initialize the complete crime detection system."""
        try:
            logger.info("Initializing Crime Detection System...")
            
            # 1. Initialize database
            logger.info("Setting up database...")
            db_config = DatabaseConfig.from_env()
            
            # Validate configuration
            errors = db_config.validate()
            if errors:
                logger.error(f"Database configuration errors: {errors}")
                return False
            
            # Initialize database schema
            success = initialize_database(db_config.database_path)
            if not success:
                logger.error("Failed to initialize database schema")
                return False
            
            # Initialize connection manager
            conn_manager = initialize_database_connection(db_config)
            
            # Test database connection
            if not conn_manager.test_connection():
                logger.error("Database connection test failed")
                return False
            
            logger.info("Database initialized successfully")
            
            # 2. Initialize database service
            logger.info("Setting up database service...")
            self.db_service = get_database_service(db_config.database_path)
            
            # 3. Initialize analytics service
            logger.info("Setting up analytics service...")
            db_operations = DatabaseOperations(conn_manager)
            self.analytics_service = AnalyticsService(db_operations)
            
            # 4. Initialize alert system
            logger.info("Setting up alert system...")
            alert_config = AlertConfiguration(
                telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', '7560595961:AAG5sAVZv4QaMVdjqx0KFJQqTVZVHtuvQ5E'),
                telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID', '6639956728'),
                max_alerts_per_minute=10,
                enable_rate_limiting=True,
                enable_duplicate_detection=True,
                duplicate_window_minutes=5
            )
            
            self.alert_manager = UnifiedAlertManager(alert_config, db_operations)
            
            # 5. Initialize template engine
            self.template_engine = get_template_engine()
            
            # 6. Test all connections
            logger.info("Testing system connections...")
            test_results = await self.test_system_connections()
            
            if not all(test_results.values()):
                logger.warning(f"Some connections failed: {test_results}")
            
            logger.info("Crime Detection System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def test_system_connections(self) -> Dict[str, bool]:
        """Test all system connections."""
        results = {}
        
        try:
            # Test database
            results['database'] = self.db_service.test_connections()['database']
            
            # Test alert system
            alert_tests = self.alert_manager.test_all_connections()
            results['telegram'] = alert_tests.get(AlertType.TELEGRAM, (False, ""))[0]
            
            # Test analytics (basic functionality)
            try:
                health_summary = self.db_service.get_system_health()
                results['analytics'] = 'error' not in health_summary
            except Exception as e:
                logger.error(f"Analytics test failed: {e}")
                results['analytics'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Connection tests failed: {e}")
            return {'database': False, 'telegram': False, 'analytics': False}
    
    async def demonstrate_anpr_detection(self):
        """Demonstrate ANPR detection logging and alerting."""
        logger.info("=== ANPR Detection Demo ===")
        
        try:
            # Simulate ANPR detection data
            detection_data = {
                'plate_number': 'ABC123',
                'confidence': 0.95,
                'location': 'Main_Gate',
                'image_path': '/path/to/detection.jpg',
                'processing_time': 0.250,
                'is_red_listed': False,
                'plate_confidence': 0.95,
                'ocr_text_raw': 'ABC 123',
                'region': 'generic',
                'country_code': 'XX',
                'bounding_box': {'x1': 100, 'y1': 50, 'x2': 200, 'y2': 100},
                'metadata': {
                    'camera_id': 'cam_001',
                    'weather': 'clear',
                    'lighting': 'good'
                }
            }
            
            # Log detection to database
            event_id = self.db_service.log_detection_event('ANPR', detection_data)
            
            if event_id:
                logger.info(f"ANPR detection logged with Event ID: {event_id}")
            else:
                logger.error("Failed to log ANPR detection")
                return
            
            # Now simulate a red-listed vehicle detection
            logger.info("Adding vehicle to red-list and simulating detection...")
            
            # Add vehicle to red-list
            success = self.db_service.add_red_listed_item(
                'VEHICLE', 'XYZ789', 'Stolen Vehicle', 'HIGH'
            )
            
            if success:
                logger.info("Vehicle added to red-list successfully")
                
                # Simulate detection of red-listed vehicle
                red_listed_data = {
                    **detection_data,
                    'plate_number': 'XYZ789',
                    'is_red_listed': True,
                    'alert_reason': 'Stolen Vehicle'
                }
                
                red_event_id = self.db_service.log_detection_event('ANPR', red_listed_data)
                
                if red_event_id:
                    logger.info(f"Red-listed vehicle detection logged with Event ID: {red_event_id}")
                    logger.info("This should trigger a Telegram alert")
            
        except Exception as e:
            logger.error(f"ANPR demo failed: {e}")
    
    async def demonstrate_face_detection(self):
        """Demonstrate face detection logging and alerting."""
        logger.info("=== Face Detection Demo ===")
        
        try:
            # Simulate unknown person detection
            face_data = {
                'confidence': 0.88,
                'location': 'Entrance_A',
                'image_path': '/path/to/face.jpg',
                'processing_time': 0.150,
                'person_name': None,
                'person_category': 'unknown',
                'recognition_confidence': 0.88,
                'age_estimate': 35,
                'gender_estimate': 'male',
                'emotion_detected': 'neutral',
                'face_quality_score': 0.9,
                'is_masked': False,
                'bounding_box': {'x1': 150, 'y1': 100, 'x2': 250, 'y2': 200},
                'metadata': {
                    'detection_model': 'face_recognition_v2',
                    'lighting_quality': 'good'
                }
            }
            
            event_id = self.db_service.log_detection_event('FACE', face_data)
            
            if event_id:
                logger.info(f"Unknown face detection logged with Event ID: {event_id}")
            
            # Simulate restricted person detection
            logger.info("Simulating restricted person detection...")
            
            restricted_data = {
                **face_data,
                'person_name': 'John Restricted',
                'person_category': 'restricted',
                'recognition_confidence': 0.92
            }
            
            restricted_event_id = self.db_service.log_detection_event('FACE', restricted_data)
            
            if restricted_event_id:
                logger.info(f"Restricted person detection logged with Event ID: {restricted_event_id}")
                logger.info("This should trigger a security alert")
            
        except Exception as e:
            logger.error(f"Face detection demo failed: {e}")
    
    async def demonstrate_violence_detection(self):
        """Demonstrate violence detection logging and alerting."""
        logger.info("=== Violence Detection Demo ===")
        
        try:
            violence_data = {
                'confidence': 0.89,
                'location': 'Parking_Lot',
                'image_path': '/path/to/violence.jpg',
                'processing_time': 0.320,
                'is_violence': True,
                'violence_type': 'physical_altercation',
                'violence_intensity': 'HIGH',
                'duration_seconds': 15.5,
                'people_count': 3,
                'movement_intensity': 0.85,
                'frame_analysis': {
                    'frames_analyzed': 45,
                    'violence_frames': 28,
                    'confidence_scores': [0.87, 0.89, 0.91, 0.88]
                },
                'metadata': {
                    'detection_algorithm': 'violence_cnn_v1',
                    'camera_angle': 'overhead'
                }
            }
            
            event_id = self.db_service.log_detection_event('VIOLENCE', violence_data)
            
            if event_id:
                logger.info(f"Violence detection logged with Event ID: {event_id}")
                logger.info("This should trigger an emergency alert")
            
        except Exception as e:
            logger.error(f"Violence detection demo failed: {e}")
    
    async def demonstrate_weapon_detection(self):
        """Demonstrate weapon detection logging and alerting."""
        logger.info("=== Weapon Detection Demo ===")
        
        try:
            weapon_data = {
                'confidence': 0.94,
                'location': 'Security_Checkpoint',
                'image_path': '/path/to/weapon.jpg',
                'processing_time': 0.180,
                'weapon_type': 'handgun',
                'weapon_category': 'firearm',
                'threat_level': 'CRITICAL',
                'weapon_size_estimate': 'medium',
                'is_concealed': False,
                'person_holding_weapon': True,
                'weapon_condition': 'drawn',
                'multiple_weapons': False,
                'weapon_count': 1,
                'bounding_box': {'x1': 300, 'y1': 200, 'x2': 350, 'y2': 250},
                'metadata': {
                    'yolo_model': 'weapon_detection_v3',
                    'detection_classes': ['handgun']
                }
            }
            
            event_id = self.db_service.log_detection_event('WEAPON', weapon_data)
            
            if event_id:
                logger.info(f"Weapon detection logged with Event ID: {event_id}")
                logger.info("This should trigger a critical threat alert")
            
        except Exception as e:
            logger.error(f"Weapon detection demo failed: {e}")
    
    async def demonstrate_analytics_features(self):
        """Demonstrate analytics and search capabilities."""
        logger.info("=== Analytics Demo ===")
        
        try:
            # Get dashboard overview
            logger.info("Getting dashboard overview...")
            overview = self.analytics_service.get_dashboard_overview(TimeRange.DAY)
            
            if 'error' not in overview:
                logger.info(f"Dashboard Overview:")
                logger.info(f"  Total Events: {overview['summary']['total_events']}")
                logger.info(f"  Alert Events: {overview['summary']['alert_events']}")
                logger.info(f"  Critical Events: {overview['summary']['critical_events']}")
                logger.info(f"  Detection Counts: {overview['detection_counts']}")
            
            # Get real-time metrics
            logger.info("Getting real-time metrics...")
            realtime = self.analytics_service.get_real_time_metrics()
            
            if 'error' not in realtime:
                logger.info(f"Real-time Metrics:")
                logger.info(f"  Events Last Hour: {realtime['activity']['events_last_hour']}")
                logger.info(f"  Alerts Last Hour: {realtime['activity']['alerts_last_hour']}")
                logger.info(f"  Avg Processing Time: {realtime['system_health']['avg_processing_time_ms']:.2f}ms")
            
            # Demonstrate search functionality
            logger.info("Searching for recent critical events...")
            search_query = SearchQuery(
                filters=[
                    SearchFilter(field='severity_level', operator='eq', value='CRITICAL'),
                    SearchFilter(field='timestamp', operator='gte', value=datetime.now() - timedelta(hours=24))
                ],
                limit=10
            )
            
            search_results = self.analytics_service.search_events(search_query)
            logger.info(f"Found {search_results['total_count']} critical events in the last 24 hours")
            
            # Get module-specific analytics
            for module in ['ANPR', 'FACE', 'VIOLENCE', 'WEAPON']:
                logger.info(f"Getting {module} analytics...")
                module_analytics = self.analytics_service.get_module_analytics(module, TimeRange.DAY)
                
                if 'error' not in module_analytics:
                    logger.info(f"  {module} - Recent activity available")
            
        except Exception as e:
            logger.error(f"Analytics demo failed: {e}")
    
    async def demonstrate_alert_system(self):
        """Demonstrate alert system capabilities."""
        logger.info("=== Alert System Demo ===")
        
        try:
            # Send test system alert
            logger.info("Sending test system alert...")
            success = self.alert_manager.send_system_alert(
                "🧪 <b>System Test Alert</b>\n\nThis is a test of the unified alert system. All components are functioning correctly.",
                AlertPriority.LOW
            )
            
            if success:
                logger.info("Test alert sent successfully")
            else:
                logger.error("Failed to send test alert")
            
            # Get alert service statistics
            stats = self.alert_manager.get_service_stats()
            logger.info(f"Alert Service Stats:")
            logger.info(f"  Queue Size: {stats['telegram']['queue_size']}")
            logger.info(f"  Bot Initialized: {stats['telegram']['bot_initialized']}")
            logger.info(f"  Processing Thread Active: {stats['telegram']['processing_thread_active']}")
            
        except Exception as e:
            logger.error(f"Alert system demo failed: {e}")
    
    async def demonstrate_system_management(self):
        """Demonstrate system management features."""
        logger.info("=== System Management Demo ===")
        
        try:
            # Get system health
            logger.info("Checking system health...")
            health = self.db_service.get_system_health()
            
            if 'error' not in health:
                logger.info("System Health Check:")
                logger.info(f"  Database Connection: Active")
                logger.info(f"  Database Size: {health['database']['db_size_mb']:.2f} MB")
                logger.info(f"  Total Logs (24h): {health['total_logs_24h']}")
            
            # Get red-listed items
            logger.info("Getting red-listed items...")
            red_items = self.db_service.get_red_listed_items()
            logger.info(f"Currently {len(red_items)} items on red-list")
            
            for item in red_items[:3]:  # Show first 3
                logger.info(f"  - {item['type']}: {item['identifier']} ({item['reason']})")
            
        except Exception as e:
            logger.error(f"System management demo failed: {e}")
    
    async def run_complete_demo(self):
        """Run the complete system demonstration."""
        logger.info("Starting Complete Crime Detection System Demo")
        logger.info("=" * 60)
        
        try:
            # Initialize system
            if not await self.initialize_system():
                logger.error("System initialization failed - aborting demo")
                return
            
            # Run detection demos
            await self.demonstrate_anpr_detection()
            await asyncio.sleep(2)  # Brief pause between demos
            
            await self.demonstrate_face_detection()
            await asyncio.sleep(2)
            
            await self.demonstrate_violence_detection()
            await asyncio.sleep(2)
            
            await self.demonstrate_weapon_detection()
            await asyncio.sleep(2)
            
            # Run analytics demo
            await self.demonstrate_analytics_features()
            await asyncio.sleep(2)
            
            # Run alert system demo
            await self.demonstrate_alert_system()
            await asyncio.sleep(2)
            
            # Run system management demo
            await self.demonstrate_system_management()
            
            logger.info("=" * 60)
            logger.info("Complete system demonstration finished successfully!")
            logger.info("Check your Telegram for alert notifications")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
    
    def shutdown(self):
        """Shutdown the system gracefully."""
        logger.info("Shutting down Crime Detection System...")
        
        try:
            if self.db_service:
                self.db_service.shutdown()
            
            if self.alert_manager:
                self.alert_manager.shutdown()
            
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

async def main():
    """Main function to run the demo."""
    demo = CrimeDetectionSystemExample()
    
    try:
        await demo.run_complete_demo()
    finally:
        demo.shutdown()

if __name__ == "__main__":
    # Set environment variables for demo (replace with your actual values)
    os.environ.setdefault('TELEGRAM_BOT_TOKEN', '7560595961:AAG5sAVZv4QaMVdjqx0KFJQqTVZVHtuvQ5E')
    os.environ.setdefault('TELEGRAM_CHAT_ID', '6639956728')
    os.environ.setdefault('DATABASE_PATH', 'crime_detection_demo.db')
    
    # Run the demo
    asyncio.run(main())