import asyncio
import io
import logging
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError
import cv2
from PIL import Image
import numpy as np
from config import ANPRConfig

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramAlertBot:
    def __init__(self):
        self.config = ANPRConfig()
        self.bot_token = self.config.telegram.bot_token if hasattr(self.config.telegram, 'bot_token') else 'YOUR_BOT_TOKEN_HERE'
        self.chat_id = self.config.telegram.chat_id if hasattr(self.config.telegram, 'chat_id') else ''
        self.bot = None
        self.initialize_bot()
    
    def initialize_bot(self):
        """Initialize the Telegram bot."""
        try:
            if self.bot_token == 'YOUR_BOT_TOKEN_HERE':
                logger.warning("⚠️ Telegram bot token not configured. Please update .env file.")
                return
            
            self.bot = Bot(token=self.bot_token)
            logger.info("✅ Telegram bot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Telegram bot: {e}")
    
    async def send_message(self, message):
        """Send a text message to the configured chat."""
        if not self.bot:
            logger.warning("⚠️ Telegram bot not initialized")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.info("✅ Message sent successfully")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Error sending message: {e}")
            return False
    
    async def send_photo_with_message(self, image, message):
        """Send a photo with a message to the configured chat."""
        if not self.bot:
            logger.warning("⚠️ Telegram bot not initialized")
            return False
        
        try:
            # Convert OpenCV image to bytes
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=img_byte_arr,
                    caption=message,
                    parse_mode='HTML'
                )
                
                logger.info("✅ Photo with message sent successfully")
                return True
            else:
                logger.error("❌ Invalid image format")
                return False
                
        except TelegramError as e:
            logger.error(f"❌ Error sending photo: {e}")
            return False
    
    async def send_unknown_person_alert(self, image, timestamp=None, location="Camera 1"):
        """Send an alert for an unknown person detection."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format the alert message
        message = f"""
🚨 <b>UNKNOWN PERSON DETECTED</b> 🚨

📅 <b>Date:</b> {timestamp.strftime('%Y-%m-%d')}
🕒 <b>Time:</b> {timestamp.strftime('%H:%M:%S')}
📍 <b>Location:</b> {location}

⚠️ An unidentified person has been detected in the surveillance area. Please review the attached image and take appropriate action if necessary.

#UnknownPerson #SecurityAlert
        """
        
        return await self.send_photo_with_message(image, message.strip())
    
    async def send_system_status(self, stats):
        """Send system status and statistics."""
        message = f"""
📊 <b>Face Detection System Status</b>

🔍 <b>Total Detections:</b> {stats.get('total_detections', 0)}
✅ <b>Known Persons:</b> {stats.get('known_detections', 0)}
❌ <b>Unknown Persons:</b> {stats.get('unknown_detections', 0)}
🚨 <b>Recent Alerts (24h):</b> {stats.get('recent_alerts', 0)}

🟢 <b>System Status:</b> Running
📅 <b>Last Update:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#SystemStatus #Statistics
        """
        
        return await self.send_message(message.strip())
    
    async def send_red_plate_alert(self, image, plate_number, alert_reason, timestamp=None, location="Camera 1"):
        """Send an alert for a red-listed vehicle detection."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format the alert message
        message = f"""
🚨 <b>RED ALERT: SUSPICIOUS VEHICLE DETECTED</b> 🚨

🚗 <b>Plate Number:</b> <code>{plate_number}</code>
📅 <b>Date:</b> {timestamp.strftime('%Y-%m-%d')}
🕒 <b>Time:</b> {timestamp.strftime('%H:%M:%S')}
📍 <b>Location:</b> {location}
⚠️ <b>Alert Reason:</b> {alert_reason}

🔴 This vehicle is on the red alert list. Please take immediate action and verify the situation.

#RedAlert #SuspiciousVehicle #PlateRecognition
        """
        
        return await self.send_photo_with_message(image, message.strip())
    
    async def send_plate_detection_status(self, stats):
        """Send plate detection system status and statistics."""
        message = f"""
📊 <b>Plate Recognition System Status</b>

🔍 <b>Total Plate Detections:</b> {stats.get('total_plate_detections', 0)}
✅ <b>Normal Vehicles:</b> {stats.get('normal_detections', 0)}
🚨 <b>Red-Listed Vehicles:</b> {stats.get('red_listed_detections', 0)}
⚠️ <b>Recent Alerts (24h):</b> {stats.get('recent_red_alerts', 0)}
📋 <b>Total Red-Listed Vehicles:</b> {stats.get('total_red_vehicles', 0)}

🟢 <b>System Status:</b> Running
📅 <b>Last Update:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#PlateRecognition #SystemStatus #Statistics
        """
        
        return await self.send_message(message.strip())
    
    async def send_startup_message(self):
        """Send a message when the system starts up."""
        message = f"""
🚀 <b>Face Detection System Started</b>

✅ System is now online and monitoring for unknown persons.
📱 You will receive alerts when unidentified individuals are detected.

🕒 <b>Started at:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#SystemStartup #Online
        """
        
        return await self.send_message(message.strip())
    
    async def send_shutdown_message(self):
        """Send a message when the system shuts down."""
        message = f"""
🔴 <b>Face Detection System Stopped</b>

⚠️ System has been stopped and is no longer monitoring.

🕒 <b>Stopped at:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#SystemShutdown #Offline
        """
        
        return await self.send_message(message.strip())
    
    async def test_connection(self):
        """Test the Telegram bot connection."""
        try:
            if not self.bot:
                return False, "Bot not initialized"
            
            # Get bot info
            bot_info = await self.bot.get_me()
            logger.info(f"✅ Bot connection test successful. Bot name: {bot_info.first_name}")
            
            # Send test message
            test_message = f"""
🧪 <b>Connection Test</b>

✅ Telegram bot is working correctly!
🤖 <b>Bot Name:</b> {bot_info.first_name}
📅 <b>Test Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#ConnectionTest #Success
            """
            
            success = await self.send_message(test_message.strip())
            return success, "Connection test successful"
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False, str(e)

# Utility functions for running async functions
def run_async(coro):
    """Run an async function in a synchronous context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def send_alert_sync(image, timestamp=None, location="Camera 1"):
    """Synchronous wrapper for sending alerts."""
    bot = TelegramAlertBot()
    return run_async(bot.send_unknown_person_alert(image, timestamp, location))

def send_message_sync(message):
    """Synchronous wrapper for sending messages."""
    bot = TelegramAlertBot()
    return run_async(bot.send_message(message))

def test_telegram_connection():
    """Test Telegram bot connection synchronously."""
    bot = TelegramAlertBot()
    return run_async(bot.test_connection())