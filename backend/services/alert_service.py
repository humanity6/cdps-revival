"""
Unified Telegram Alert Service for Real-Time Crime Detection System
Handles alerts for all detection modules with intelligent prioritization and rate limiting
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
import json
import hashlib
from enum import Enum
import os

try:
    from telegram import Bot
    from telegram.error import TelegramError, NetworkError, RetryAfter
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    TelegramError = Exception
    NetworkError = Exception
    RetryAfter = Exception

from ..database.operations import DatabaseOperations
from ..database.models import DatabaseManager, Event, EventTypeEnum, SeverityLevelEnum

logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Alert priority levels for queue processing."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

class AlertType(Enum):
    """Types of alerts supported."""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SYSTEM = "system"

@dataclass
class AlertConfiguration:
    """Configuration for alert system."""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    max_alerts_per_minute: int = 10
    alert_cooldown_seconds: int = 60
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_rate_limiting: bool = True
    enable_duplicate_detection: bool = True
    duplicate_window_minutes: int = 5
    batch_processing: bool = False
    batch_size: int = 5
    batch_delay_seconds: int = 2

@dataclass
class AlertMessage:
    """Represents an alert message in the queue."""
    priority: AlertPriority
    event_id: int
    alert_type: AlertType
    recipient: str
    subject: str
    content: str
    template_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value

class TelegramAlertService:
    """Enhanced Telegram alert service with intelligent features."""
    
    def __init__(self, config: AlertConfiguration, db_operations: DatabaseOperations):
        self.config = config
        self.db_ops = db_operations
        self.bot = None
        self.alert_queue = PriorityQueue()
        self.rate_limiter = {}
        self.duplicate_tracker = {}
        self.processing_thread = None
        self.is_running = False
        
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram package not available. Install python-telegram-bot to enable Telegram alerts.")
            return
        
        self._initialize_bot()
        self._start_processing_thread()
    
    def _initialize_bot(self):
        """Initialize the Telegram bot."""
        if not self.config.telegram_bot_token or not TELEGRAM_AVAILABLE:
            logger.warning("Telegram bot token not configured or Telegram package not available")
            return
        
        try:
            self.bot = Bot(token=self.config.telegram_bot_token)
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
    
    def _start_processing_thread(self):
        """Start the background thread for processing alerts."""
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_alert_queue, daemon=True)
            self.processing_thread.start()
            logger.info("Alert processing thread started")
    
    def _stop_processing_thread(self):
        """Stop the background processing thread."""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
            logger.info("Alert processing thread stopped")
    
    def _process_alert_queue(self):
        """Background thread to process alert queue."""
        while self.is_running:
            try:
                # Get alert from queue with timeout
                try:
                    alert = self.alert_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Check rate limiting
                if self._is_rate_limited(alert):
                    # Put back in queue with delay
                    time.sleep(1)
                    self.alert_queue.put(alert)
                    continue
                
                # Process the alert
                success = self._send_alert(alert)
                
                # Update database
                status = 'SENT' if success else 'FAILED'
                error_msg = None if success else f"Failed after {alert.retry_count} attempts"
                
                self.db_ops.update_alert_status(
                    alert_id=alert.metadata.get('alert_id'),
                    status=status,
                    error_message=error_msg
                )
                
                # Handle retry logic
                if not success and alert.retry_count < alert.max_retries:
                    alert.retry_count += 1
                    # Exponential backoff
                    delay = self.config.retry_delay_seconds * (2 ** alert.retry_count)
                    time.sleep(min(delay, 60))  # Cap at 60 seconds
                    self.alert_queue.put(alert)
                    logger.info(f"Retrying alert {alert.metadata.get('alert_id')} (attempt {alert.retry_count})")
                
                self.alert_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in alert processing thread: {e}")
                time.sleep(1)
    
    def _is_rate_limited(self, alert: AlertMessage) -> bool:
        """Check if alert should be rate limited."""
        if not self.config.enable_rate_limiting:
            return False
        
        now = datetime.now()
        minute_key = f"{alert.recipient}_{now.strftime('%Y%m%d%H%M')}"
        
        if minute_key not in self.rate_limiter:
            self.rate_limiter[minute_key] = 0
        
        # Clean old entries
        self._cleanup_rate_limiter()
        
        if self.rate_limiter[minute_key] >= self.config.max_alerts_per_minute:
            return True
        
        self.rate_limiter[minute_key] += 1
        return False
    
    def _cleanup_rate_limiter(self):
        """Clean up old rate limiter entries."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=2)
        
        keys_to_remove = []
        for key in self.rate_limiter:
            try:
                key_time = datetime.strptime(key.split('_', 1)[1], '%Y%m%d%H%M')
                if key_time < cutoff:
                    keys_to_remove.append(key)
            except (ValueError, IndexError):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.rate_limiter[key]
    
    def _is_duplicate_alert(self, alert: AlertMessage) -> bool:
        """Check if this is a duplicate alert within the time window."""
        if not self.config.enable_duplicate_detection:
            return False
        
        # Create hash of alert content for duplicate detection
        content_hash = hashlib.md5(
            f"{alert.template_name}_{alert.recipient}_{alert.content}".encode()
        ).hexdigest()
        
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.config.duplicate_window_minutes)
        
        # Clean old entries
        keys_to_remove = [
            key for key, timestamp in self.duplicate_tracker.items()
            if timestamp < cutoff
        ]
        for key in keys_to_remove:
            del self.duplicate_tracker[key]
        
        if content_hash in self.duplicate_tracker:
            logger.info(f"Skipping duplicate alert: {alert.template_name}")
            return True
        
        self.duplicate_tracker[content_hash] = now
        return False
    
    def _send_alert(self, alert: AlertMessage) -> bool:
        """Send individual alert based on type."""
        try:
            if alert.alert_type == AlertType.TELEGRAM:
                return self._send_telegram_alert(alert)
            elif alert.alert_type == AlertType.EMAIL:
                return self._send_email_alert(alert)
            elif alert.alert_type == AlertType.SMS:
                return self._send_sms_alert(alert)
            elif alert.alert_type == AlertType.WEBHOOK:
                return self._send_webhook_alert(alert)
            else:
                logger.warning(f"Unsupported alert type: {alert.alert_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    def _send_telegram_alert(self, alert: AlertMessage) -> bool:
        """Send Telegram alert with image support."""
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return False
        
        try:
            # Create event loop for async operations
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self._send_telegram_message_async(alert))
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    async def _send_telegram_message_async(self, alert: AlertMessage) -> bool:
        """Send Telegram message asynchronously."""
        try:
            if alert.image_path and os.path.exists(alert.image_path):
                # Send photo with caption
                with open(alert.image_path, 'rb') as photo:
                    await self.bot.send_photo(
                        chat_id=alert.recipient,
                        photo=photo,
                        caption=alert.content,
                        parse_mode='HTML'
                    )
            else:
                # Send text message
                await self.bot.send_message(
                    chat_id=alert.recipient,
                    text=alert.content,
                    parse_mode='HTML'
                )
            
            logger.info(f"Telegram alert sent successfully to {alert.recipient}")
            return True
            
        except RetryAfter as e:
            logger.warning(f"Rate limited by Telegram, retry after {e.retry_after} seconds")
            time.sleep(e.retry_after)
            return False
        except NetworkError as e:
            logger.error(f"Network error sending Telegram alert: {e}")
            return False
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram alert: {e}")
            return False
    
    def _send_email_alert(self, alert: AlertMessage) -> bool:
        """Send email alert (placeholder for future implementation)."""
        logger.info(f"Email alert would be sent to {alert.recipient}: {alert.subject}")
        return True
    
    def _send_sms_alert(self, alert: AlertMessage) -> bool:
        """Send SMS alert (placeholder for future implementation)."""
        logger.info(f"SMS alert would be sent to {alert.recipient}: {alert.content[:100]}")
        return True
    
    def _send_webhook_alert(self, alert: AlertMessage) -> bool:
        """Send webhook alert (placeholder for future implementation)."""
        logger.info(f"Webhook alert would be sent to {alert.recipient}")
        return True
    
    def queue_alert(self, alert: AlertMessage) -> bool:
        """Add alert to processing queue."""
        try:
            # Check for duplicates
            if self._is_duplicate_alert(alert):
                return False
            
            # Create database record
            alert_id = self.db_ops.create_alert({
                'event_id': alert.event_id,
                'alert_type': alert.alert_type.value,
                'recipient': alert.recipient,
                'message_content': alert.content,
                'message_template': alert.template_name,
                'priority': alert.priority.value
            })
            
            if alert_id:
                alert.metadata['alert_id'] = alert_id
                self.alert_queue.put(alert)
                logger.info(f"Alert queued: {alert.template_name} for event {alert.event_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to queue alert: {e}")
            return False
    
    def send_immediate_alert(self, alert: AlertMessage) -> bool:
        """Send alert immediately without queuing."""
        try:
            if self._is_duplicate_alert(alert):
                return False
            
            # Create database record
            alert_id = self.db_ops.create_alert({
                'event_id': alert.event_id,
                'alert_type': alert.alert_type.value,
                'recipient': alert.recipient,
                'message_content': alert.content,
                'message_template': alert.template_name,
                'priority': alert.priority.value
            })
            
            if alert_id:
                alert.metadata['alert_id'] = alert_id
                success = self._send_alert(alert)
                
                # Update database status
                status = 'SENT' if success else 'FAILED'
                self.db_ops.update_alert_status(alert_id, status)
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send immediate alert: {e}")
            return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.alert_queue.qsize()
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert service statistics."""
        return {
            'queue_size': self.get_queue_size(),
            'rate_limiter_entries': len(self.rate_limiter),
            'duplicate_tracker_entries': len(self.duplicate_tracker),
            'bot_initialized': self.bot is not None,
            'processing_thread_active': self.processing_thread and self.processing_thread.is_alive(),
            'telegram_available': TELEGRAM_AVAILABLE
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Telegram bot connection."""
        if not self.bot:
            return False, "Bot not initialized"
        
        try:
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def test_bot():
                try:
                    bot_info = await self.bot.get_me()
                    return True, f"Connected as {bot_info.first_name}"
                except Exception as e:
                    return False, str(e)
            
            return loop.run_until_complete(test_bot())
            
        except Exception as e:
            return False, str(e)
    
    def shutdown(self):
        """Shutdown the alert service."""
        logger.info("Shutting down alert service...")
        self._stop_processing_thread()
        
        # Process remaining alerts in queue
        remaining_alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                remaining_alerts.append(alert)
            except Empty:
                break
        
        logger.info(f"Alert service shutdown complete. {len(remaining_alerts)} alerts remaining in queue.")

class UnifiedAlertManager:
    """Main alert manager that coordinates all alert services."""
    
    def __init__(self, config: AlertConfiguration, db_operations: DatabaseOperations):
        self.config = config
        self.db_ops = db_operations
        self.telegram_service = TelegramAlertService(config, db_operations)
        
        # Alert service registry
        self.services = {
            AlertType.TELEGRAM: self.telegram_service,
            # Add other services here as implemented
        }
    
    def send_detection_alert(self, event: Event, alert_type: AlertType = AlertType.TELEGRAM,
                           priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Send alert for a detection event."""
        try:
            # Get appropriate template and content based on event type
            template_name, content, subject = self._get_alert_content(event)
            
            # Determine recipient
            recipient = self._get_recipient(alert_type)
            if not recipient:
                logger.error(f"No recipient configured for {alert_type}")
                return False
            
            # Create alert message
            alert = AlertMessage(
                priority=priority,
                event_id=event.id,
                alert_type=alert_type,
                recipient=recipient,
                subject=subject,
                content=content,
                template_name=template_name,
                image_path=event.image_path,
                metadata={'event_type': event.event_type, 'severity': event.severity_level}
            )
            
            # Send based on priority
            if priority == AlertPriority.CRITICAL:
                return self.services[alert_type].send_immediate_alert(alert)
            else:
                return self.services[alert_type].queue_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to send detection alert: {e}")
            return False
    
    def _get_alert_content(self, event: Event) -> Tuple[str, str, str]:
        """Get alert content based on event type."""
        # This will be enhanced with proper templates
        base_time = event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        if event.event_type == 'ANPR':
            anpr_event = event.anpr_event
            if anpr_event and anpr_event.is_red_listed:
                template = "anpr_red_alert"
                subject = "🚨 RED ALERT: Suspicious Vehicle Detected"
                content = f"""
🚨 <b>RED ALERT: SUSPICIOUS VEHICLE DETECTED</b> 🚨

🚗 <b>Plate Number:</b> <code>{anpr_event.plate_number}</code>
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
⚠️ <b>Alert Reason:</b> {anpr_event.alert_reason or 'Suspicious Activity'}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

🔴 This vehicle is on the red alert list. Please take immediate action.

#RedAlert #SuspiciousVehicle #ANPR
                """.strip()
            else:
                template = "anpr_detection"
                subject = "Vehicle Detected"
                content = f"""
🚗 <b>Vehicle Detection</b>

📋 <b>Plate Number:</b> <code>{anpr_event.plate_number if anpr_event else 'Unknown'}</code>
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

#VehicleDetection #ANPR
                """.strip()
        
        elif event.event_type == 'FACE':
            face_event = event.face_event
            if face_event and face_event.person_category in ['restricted', 'criminal']:
                template = "face_alert"
                subject = "🚨 SECURITY ALERT: Restricted Person Detected"
                content = f"""
🚨 <b>SECURITY ALERT: RESTRICTED PERSON DETECTED</b> 🚨

👤 <b>Person:</b> {face_event.person_name or 'Unknown'}
📂 <b>Category:</b> {face_event.person_category.upper()}
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

🔴 Immediate security response may be required.

#SecurityAlert #FaceRecognition #RestrictedPerson
                """.strip()
            else:
                template = "face_detection"
                subject = "Unknown Person Detected"
                content = f"""
👤 <b>Unknown Person Detected</b>

📂 <b>Category:</b> {face_event.person_category if face_event else 'Unknown'}
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

#UnknownPerson #FaceRecognition
                """.strip()
        
        elif event.event_type == 'VIOLENCE':
            violence_event = event.violence_event
            template = "violence_alert"
            subject = "🚨 VIOLENCE ALERT: Violent Activity Detected"
            content = f"""
🚨 <b>VIOLENCE ALERT: VIOLENT ACTIVITY DETECTED</b> 🚨

⚡ <b>Violence Type:</b> {violence_event.violence_type if violence_event else 'Unknown'}
📊 <b>Intensity:</b> {violence_event.violence_intensity if violence_event else 'Unknown'}
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

🚨 IMMEDIATE RESPONSE REQUIRED

#ViolenceAlert #Emergency #SecurityThreat
            """.strip()
        
        elif event.event_type == 'WEAPON':
            weapon_event = event.weapon_event
            template = "weapon_alert"
            subject = "🚨 CRITICAL ALERT: Weapon Detected"
            content = f"""
🚨 <b>CRITICAL ALERT: WEAPON DETECTED</b> 🚨

🔫 <b>Weapon Type:</b> {weapon_event.weapon_type if weapon_event else 'Unknown'}
⚠️ <b>Threat Level:</b> {weapon_event.threat_level if weapon_event else 'HIGH'}
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

🚨 CRITICAL THREAT - IMMEDIATE RESPONSE REQUIRED

#WeaponAlert #CriticalThreat #Emergency
            """.strip()
        
        else:
            template = "generic_alert"
            subject = f"Detection Alert: {event.event_type}"
            content = f"""
🔍 <b>Detection Alert</b>

📋 <b>Type:</b> {event.event_type}
📅 <b>Time:</b> {base_time}
📍 <b>Location:</b> {event.location}
🎯 <b>Confidence:</b> {event.confidence_score:.2%}

#Detection #{event.event_type}
            """.strip()
        
        return template, content, subject
    
    def _get_recipient(self, alert_type: AlertType) -> Optional[str]:
        """Get recipient for alert type."""
        if alert_type == AlertType.TELEGRAM:
            return self.config.telegram_chat_id
        # Add other alert types here
        return None
    
    def send_system_alert(self, message: str, priority: AlertPriority = AlertPriority.LOW) -> bool:
        """Send system status alert."""
        try:
            alert = AlertMessage(
                priority=priority,
                event_id=0,  # System alerts don't have event ID
                alert_type=AlertType.TELEGRAM,
                recipient=self.config.telegram_chat_id,
                subject="System Alert",
                content=f"🔧 <b>System Alert</b>\n\n{message}\n\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                template_name="system_alert"
            )
            
            return self.telegram_service.queue_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
            return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics for all alert services."""
        return {
            'telegram': self.telegram_service.get_alert_stats(),
            'total_services': len(self.services),
            'active_services': sum(1 for service in self.services.values() if service.bot is not None)
        }
    
    def test_all_connections(self) -> Dict[AlertType, Tuple[bool, str]]:
        """Test connections for all alert services."""
        results = {}
        for alert_type, service in self.services.items():
            if hasattr(service, 'test_connection'):
                results[alert_type] = service.test_connection()
            else:
                results[alert_type] = (False, "Test not implemented")
        return results
    
    def shutdown(self):
        """Shutdown all alert services."""
        for service in self.services.values():
            if hasattr(service, 'shutdown'):
                service.shutdown()