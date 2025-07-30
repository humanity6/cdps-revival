"""
Alert Message Templates for Unified Crime Detection System
Provides customizable templates for different alert types and events
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Types of alert templates."""
    ANPR_RED_ALERT = "anpr_red_alert"
    ANPR_DETECTION = "anpr_detection"
    FACE_RESTRICTED_ALERT = "face_restricted_alert"
    FACE_CRIMINAL_ALERT = "face_criminal_alert"
    FACE_UNKNOWN_DETECTION = "face_unknown_detection"
    VIOLENCE_ALERT = "violence_alert"
    WEAPON_CRITICAL_ALERT = "weapon_critical_alert"
    WEAPON_HIGH_ALERT = "weapon_high_alert"
    WEAPON_DETECTION = "weapon_detection"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_STATUS = "system_status"
    BATCH_SUMMARY = "batch_summary"
    DAILY_REPORT = "daily_report"

class MessageFormat(Enum):
    """Message format types."""
    TELEGRAM_HTML = "telegram_html"
    PLAIN_TEXT = "plain_text"
    EMAIL_HTML = "email_html"
    SMS = "sms"

@dataclass
class AlertTemplate:
    """Alert template definition."""
    name: str
    template_type: TemplateType
    format_type: MessageFormat
    subject_template: str
    content_template: str
    priority_level: int  # 1-5 scale
    includes_image: bool = False
    max_length: Optional[int] = None
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class AlertTemplateEngine:
    """Template engine for generating alert messages."""
    
    def __init__(self):
        self.templates: Dict[TemplateType, AlertTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default alert templates."""
        
        # ANPR Templates
        self.templates[TemplateType.ANPR_RED_ALERT] = AlertTemplate(
            name="ANPR Red Alert",
            template_type=TemplateType.ANPR_RED_ALERT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🚨 RED ALERT: Suspicious Vehicle Detected",
            content_template="""
🚨 <b>RED ALERT: SUSPICIOUS VEHICLE DETECTED</b> 🚨

🚗 <b>Plate Number:</b> <code>{plate_number}</code>
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
⚠️ <b>Alert Reason:</b> {alert_reason}
🎯 <b>Confidence:</b> {confidence}%
🌍 <b>Region:</b> {region}

🔴 This vehicle is on the red alert list. Please take immediate action and verify the situation.

{additional_info}

#RedAlert #SuspiciousVehicle #ANPR #Priority{priority}
            """.strip(),
            priority_level=1,
            includes_image=True,
            tags=["red_alert", "anpr", "vehicle", "suspicious"],
            description="Critical alert for red-listed vehicle detection"
        )
        
        self.templates[TemplateType.ANPR_DETECTION] = AlertTemplate(
            name="ANPR Detection",
            template_type=TemplateType.ANPR_DETECTION,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🚗 Vehicle Detected",
            content_template="""
🚗 <b>Vehicle Detection</b>

📋 <b>Plate Number:</b> <code>{plate_number}</code>
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Confidence:</b> {confidence}%
🌍 <b>Region:</b> {region}
🚙 <b>Vehicle:</b> {vehicle_info}

✅ Normal detection - no alerts triggered.

#VehicleDetection #ANPR #Normal
            """.strip(),
            priority_level=4,
            includes_image=True,
            tags=["detection", "anpr", "vehicle", "normal"],
            description="Standard vehicle detection notification"
        )
        
        # Face Recognition Templates
        self.templates[TemplateType.FACE_RESTRICTED_ALERT] = AlertTemplate(
            name="Face Restricted Alert",
            template_type=TemplateType.FACE_RESTRICTED_ALERT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🚨 SECURITY ALERT: Restricted Person Detected",
            content_template="""
🚨 <b>SECURITY ALERT: RESTRICTED PERSON DETECTED</b> 🚨

👤 <b>Person:</b> {person_name}
📂 <b>Category:</b> RESTRICTED
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Recognition Confidence:</b> {confidence}%
🎭 <b>Demographics:</b> {demographics}
😷 <b>Mask Status:</b> {mask_status}

⚠️ This person has restricted access. Verify authorization and take appropriate action.

{additional_info}

#SecurityAlert #RestrictedPerson #FaceRecognition #AccessControl
            """.strip(),
            priority_level=2,
            includes_image=True,
            tags=["security_alert", "face", "restricted", "access_control"],
            description="Alert for restricted person detection"
        )
        
        self.templates[TemplateType.FACE_CRIMINAL_ALERT] = AlertTemplate(
            name="Face Criminal Alert",
            template_type=TemplateType.FACE_CRIMINAL_ALERT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🚨 CRITICAL ALERT: Criminal Detected",
            content_template="""
🚨 <b>CRITICAL ALERT: KNOWN CRIMINAL DETECTED</b> 🚨

👤 <b>Person:</b> {person_name}
📂 <b>Category:</b> CRIMINAL
🚨 <b>Threat Level:</b> HIGH
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Recognition Confidence:</b> {confidence}%
🎭 <b>Demographics:</b> {demographics}

🚨 IMMEDIATE SECURITY RESPONSE REQUIRED 🚨
Contact law enforcement immediately.

{criminal_details}

#CriticalAlert #Criminal #FaceRecognition #Emergency #LawEnforcement
            """.strip(),
            priority_level=1,
            includes_image=True,
            tags=["critical_alert", "criminal", "face", "emergency"],
            description="Critical alert for known criminal detection"
        )
        
        self.templates[TemplateType.FACE_UNKNOWN_DETECTION] = AlertTemplate(
            name="Face Unknown Detection",
            template_type=TemplateType.FACE_UNKNOWN_DETECTION,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="👤 Unknown Person Detected",
            content_template="""
👤 <b>Unknown Person Detected</b>

📂 <b>Category:</b> UNKNOWN
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Detection Confidence:</b> {confidence}%
🎭 <b>Estimated Demographics:</b> {demographics}
😷 <b>Mask Status:</b> {mask_status}
⭐ <b>Face Quality:</b> {face_quality}

ℹ️ Unidentified person detected. Review for potential security concerns.

#UnknownPerson #FaceRecognition #Monitoring
            """.strip(),
            priority_level=4,
            includes_image=True,
            tags=["detection", "face", "unknown", "monitoring"],
            description="Notification for unknown person detection"
        )
        
        # Violence Detection Templates
        self.templates[TemplateType.VIOLENCE_ALERT] = AlertTemplate(
            name="Violence Alert",
            template_type=TemplateType.VIOLENCE_ALERT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🚨 VIOLENCE ALERT: Violent Activity Detected",
            content_template="""
🚨 <b>VIOLENCE ALERT: VIOLENT ACTIVITY DETECTED</b> 🚨

⚡ <b>Violence Type:</b> {violence_type}
📊 <b>Intensity Level:</b> {violence_intensity}
👥 <b>People Involved:</b> {people_count}
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Detection Confidence:</b> {confidence}%
⏱️ <b>Duration:</b> {duration} seconds
🏃 <b>Movement Intensity:</b> {movement_intensity}

🚨 IMMEDIATE INTERVENTION REQUIRED 🚨
Security personnel should respond immediately.

{violence_analysis}

#ViolenceAlert #Emergency #SecurityThreat #ImmediateResponse
            """.strip(),
            priority_level=1,
            includes_image=True,
            tags=["violence_alert", "emergency", "security_threat"],
            description="Critical alert for violence detection"
        )
        
        # Weapon Detection Templates
        self.templates[TemplateType.WEAPON_CRITICAL_ALERT] = AlertTemplate(
            name="Weapon Critical Alert",
            template_type=TemplateType.WEAPON_CRITICAL_ALERT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🚨 CRITICAL THREAT: Weapon Detected",
            content_template="""
🚨 <b>CRITICAL THREAT: WEAPON DETECTED</b> 🚨

🔫 <b>Weapon Type:</b> {weapon_type}
📂 <b>Category:</b> {weapon_category}
⚠️ <b>Threat Level:</b> CRITICAL
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Detection Confidence:</b> {confidence}%
📏 <b>Size Estimate:</b> {weapon_size}
👤 <b>Person Holding:</b> {person_holding}
🔍 <b>Weapon Condition:</b> {weapon_condition}
🔢 <b>Weapon Count:</b> {weapon_count}

🚨 CRITICAL THREAT - LOCKDOWN PROTOCOLS ACTIVATED 🚨
Contact emergency services immediately.

{weapon_details}

#CriticalThreat #WeaponDetected #Emergency #Lockdown #911
            """.strip(),
            priority_level=1,
            includes_image=True,
            tags=["critical_threat", "weapon", "emergency", "lockdown"],
            description="Critical alert for high-threat weapon detection"
        )
        
        self.templates[TemplateType.WEAPON_HIGH_ALERT] = AlertTemplate(
            name="Weapon High Alert",
            template_type=TemplateType.WEAPON_HIGH_ALERT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="⚠️ HIGH ALERT: Weapon Detected",
            content_template="""
⚠️ <b>HIGH ALERT: WEAPON DETECTED</b> ⚠️

🔫 <b>Weapon Type:</b> {weapon_type}
📂 <b>Category:</b> {weapon_category}
⚠️ <b>Threat Level:</b> HIGH
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Detection Confidence:</b> {confidence}%
📏 <b>Size Estimate:</b> {weapon_size}
👤 <b>Person Holding:</b> {person_holding}
🔍 <b>Weapon Condition:</b> {weapon_condition}

⚠️ HIGH PRIORITY SECURITY RESPONSE REQUIRED
Increased surveillance and security measures recommended.

#HighAlert #WeaponDetected #SecurityResponse #HighPriority
            """.strip(),
            priority_level=2,
            includes_image=True,
            tags=["high_alert", "weapon", "security_response"],
            description="High priority alert for weapon detection"
        )
        
        self.templates[TemplateType.WEAPON_DETECTION] = AlertTemplate(
            name="Weapon Detection",
            template_type=TemplateType.WEAPON_DETECTION,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🔍 Weapon Detection",
            content_template="""
🔍 <b>Weapon Detection</b>

🔫 <b>Weapon Type:</b> {weapon_type}
📂 <b>Category:</b> {weapon_category}
⚠️ <b>Threat Level:</b> {threat_level}
📅 <b>Date:</b> {date}
🕒 <b>Time:</b> {time}
📍 <b>Location:</b> {location}
🎯 <b>Detection Confidence:</b> {confidence}%
📏 <b>Size Estimate:</b> {weapon_size}

ℹ️ Weapon detected with {threat_level} threat level. Monitor situation.

#WeaponDetection #Monitoring #SecurityAwareness
            """.strip(),
            priority_level=3,
            includes_image=True,
            tags=["weapon_detection", "monitoring", "security"],
            description="Standard weapon detection notification"
        )
        
        # System Templates
        self.templates[TemplateType.SYSTEM_STARTUP] = AlertTemplate(
            name="System Startup",
            template_type=TemplateType.SYSTEM_STARTUP,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🟢 System Started",
            content_template="""
🟢 <b>Crime Detection System Started</b>

🖥️ <b>System:</b> {system_name}
📅 <b>Start Time:</b> {timestamp}
🔧 <b>Version:</b> {version}
📍 <b>Modules Active:</b> {active_modules}
📊 <b>System Health:</b> {health_status}

✅ All systems operational and monitoring for threats.

#SystemStartup #Online #Operational
            """.strip(),
            priority_level=4,
            includes_image=False,
            tags=["system", "startup", "operational"],
            description="System startup notification"
        )
        
        self.templates[TemplateType.SYSTEM_SHUTDOWN] = AlertTemplate(
            name="System Shutdown",
            template_type=TemplateType.SYSTEM_SHUTDOWN,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="🔴 System Shutdown",
            content_template="""
🔴 <b>Crime Detection System Shutdown</b>

🖥️ <b>System:</b> {system_name}
📅 <b>Shutdown Time:</b> {timestamp}
⏱️ <b>Uptime:</b> {uptime}
📊 <b>Final Stats:</b> {final_stats}

⚠️ System is no longer monitoring. Manual security measures recommended.

#SystemShutdown #Offline #SecurityGap
            """.strip(),
            priority_level=3,
            includes_image=False,
            tags=["system", "shutdown", "offline"],
            description="System shutdown notification"
        )
        
        self.templates[TemplateType.SYSTEM_ERROR] = AlertTemplate(
            name="System Error",
            template_type=TemplateType.SYSTEM_ERROR,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="❌ System Error",
            content_template="""
❌ <b>System Error Detected</b>

🖥️ <b>Module:</b> {module_name}
🚨 <b>Error Level:</b> {error_level}
📅 <b>Time:</b> {timestamp}
💥 <b>Error:</b> {error_message}
🔧 <b>Function:</b> {function_name}
📍 <b>Location:</b> Line {line_number}

⚠️ System functionality may be impacted. Check system logs for details.

{error_context}

#SystemError #TechnicalIssue #MaintenanceRequired
            """.strip(),
            priority_level=2,
            includes_image=False,
            tags=["system_error", "technical", "maintenance"],
            description="System error notification"
        )
        
        self.templates[TemplateType.DAILY_REPORT] = AlertTemplate(
            name="Daily Report",
            template_type=TemplateType.DAILY_REPORT,
            format_type=MessageFormat.TELEGRAM_HTML,
            subject_template="📊 Daily Crime Detection Report",
            content_template="""
📊 <b>Daily Crime Detection Report</b>

📅 <b>Date:</b> {report_date}
🕐 <b>Reporting Period:</b> {time_period}

<b>🔍 Detection Summary:</b>
🚗 ANPR Detections: {anpr_count} ({anpr_alerts} alerts)
👤 Face Detections: {face_count} ({face_alerts} alerts)
⚡ Violence Incidents: {violence_count}
🔫 Weapon Detections: {weapon_count}

<b>🚨 Alert Summary:</b>
🔴 Critical Alerts: {critical_alerts}
🟠 High Priority: {high_alerts}
🟡 Medium Priority: {medium_alerts}
🟢 Low Priority: {low_alerts}

<b>📈 System Performance:</b>
⏱️ Average Response Time: {avg_response_time}ms
✅ System Uptime: {uptime_percentage}%
📊 Processing Efficiency: {efficiency}%

<b>🏆 Top Locations:</b>
{top_locations}

#DailyReport #Statistics #SystemHealth
            """.strip(),
            priority_level=5,
            includes_image=False,
            tags=["daily_report", "statistics", "summary"],
            description="Daily activity and statistics report"
        )
    
    def get_template(self, template_type: TemplateType) -> Optional[AlertTemplate]:
        """Get template by type."""
        return self.templates.get(template_type)
    
    def render_template(self, template_type: TemplateType, variables: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Render template with provided variables."""
        template = self.get_template(template_type)
        if not template:
            logger.error(f"Template not found: {template_type}")
            return None
        
        try:
            # Add default variables if not provided
            default_vars = self._get_default_variables()
            all_vars = {**default_vars, **variables}
            
            # Format subject and content
            subject = template.subject_template.format(**all_vars)
            content = template.content_template.format(**all_vars)
            
            # Apply length limits if specified
            if template.max_length and len(content) > template.max_length:
                content = content[:template.max_length - 3] + "..."
            
            return {
                'subject': subject,
                'content': content,
                'template_name': template.name,
                'priority_level': template.priority_level,
                'includes_image': template.includes_image,
                'tags': template.tags
            }
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return None
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            return None
    
    def _get_default_variables(self) -> Dict[str, Any]:
        """Get default template variables."""
        now = datetime.now()
        return {
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'system_name': 'Crime Detection System',
            'version': '1.0.0',
            'confidence': '0',
            'location': 'Unknown',
            'alert_reason': 'Security Concern',
            'region': 'Unknown',
            'vehicle_info': 'Unknown',
            'person_name': 'Unknown',
            'demographics': 'Unknown',
            'mask_status': 'Unknown',
            'face_quality': 'Unknown',
            'violence_type': 'Unknown',
            'violence_intensity': 'Unknown',
            'people_count': '0',
            'duration': '0',
            'movement_intensity': 'Unknown',
            'weapon_type': 'Unknown',
            'weapon_category': 'Unknown',
            'threat_level': 'MEDIUM',
            'weapon_size': 'Unknown',
            'person_holding': 'Unknown',
            'weapon_condition': 'Unknown',
            'weapon_count': '1',
            'module_name': 'Unknown',
            'error_level': 'ERROR',
            'error_message': 'Unknown error',
            'function_name': 'Unknown',
            'line_number': '0',
            'additional_info': '',
            'criminal_details': '',
            'violence_analysis': '',
            'weapon_details': '',
            'error_context': ''
        }
    
    def get_template_for_event(self, event_type: str, event_data: Dict[str, Any]) -> Optional[TemplateType]:
        """Get appropriate template type for an event."""
        if event_type == 'ANPR':
            if event_data.get('is_red_listed', False):
                return TemplateType.ANPR_RED_ALERT
            else:
                return TemplateType.ANPR_DETECTION
        
        elif event_type == 'FACE':
            category = event_data.get('person_category', 'unknown')
            if category == 'criminal':
                return TemplateType.FACE_CRIMINAL_ALERT
            elif category == 'restricted':
                return TemplateType.FACE_RESTRICTED_ALERT
            else:
                return TemplateType.FACE_UNKNOWN_DETECTION
        
        elif event_type == 'VIOLENCE':
            return TemplateType.VIOLENCE_ALERT
        
        elif event_type == 'WEAPON':
            threat_level = event_data.get('threat_level', 'MEDIUM')
            if threat_level == 'CRITICAL':
                return TemplateType.WEAPON_CRITICAL_ALERT
            elif threat_level == 'HIGH':
                return TemplateType.WEAPON_HIGH_ALERT
            else:
                return TemplateType.WEAPON_DETECTION
        
        return None
    
    def add_custom_template(self, template: AlertTemplate):
        """Add a custom template."""
        self.templates[template.template_type] = template
        logger.info(f"Added custom template: {template.name}")
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """List all available templates."""
        return {
            template_type.value: {
                'name': template.name,
                'description': template.description,
                'priority_level': template.priority_level,
                'includes_image': template.includes_image,
                'tags': template.tags,
                'format_type': template.format_type.value
            }
            for template_type, template in self.templates.items()
        }
    
    def get_templates_by_priority(self, priority_level: int) -> List[AlertTemplate]:
        """Get templates by priority level."""
        return [
            template for template in self.templates.values()
            if template.priority_level == priority_level
        ]
    
    def get_templates_by_tag(self, tag: str) -> List[AlertTemplate]:
        """Get templates containing a specific tag."""
        return [
            template for template in self.templates.values()
            if tag in template.tags
        ]

# Global template engine instance
template_engine = AlertTemplateEngine()

def get_template_engine() -> AlertTemplateEngine:
    """Get the global template engine instance."""
    return template_engine