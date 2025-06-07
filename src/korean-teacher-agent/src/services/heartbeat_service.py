import os
import logging
import aiohttp
import uuid
from typing import Optional
from src.database.homework import Homework

logger = logging.getLogger(__name__)

async def send_homework_to_heartbeat(homework: Homework, channel_id: str, from_uuid: Optional[str] = None) -> Optional[dict]:
    """
    Send homework to Heartbeat channel asynchronously
    
    Args:
        homework (Homework): The homework object to send
        channel_id (str): The Heartbeat channel ID to send to
        from_uuid (Optional[str]): UUID of the sender. If not provided, a new UUID will be generated
        
    Returns:
        Optional[str]: The response text from Heartbeat API if successful, None if failed
    """
    api_key = os.getenv("HEARTBEAT_API_KEY")
    if not api_key:
        logger.error("HEARTBEAT_API_KEY environment variable not found")
        return None
    
    logger.info(f"Sending homework {homework.id} to Heartbeat channel {channel_id}")
    logger.info(f"api_key: {api_key}")
    url = "https://api.heartbeat.chat/v0/threads"
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Generate UUID if not provided
    sender_uuid = from_uuid or str(uuid.uuid4())
    logger.info(f"sender_uuid: {sender_uuid}")
    logger.info(f"detailed_homework: {homework.detailed_homework}, type: {type(homework.detailed_homework)}")
    
    # Prepare the message content with rich text
    message_content = {
        "text": homework.detailed_homework,
        "channelID": channel_id,
        "userID": sender_uuid,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=headers, json=message_content) as response:
                response.raise_for_status()  # Raise an exception for bad status codes
                response_data = await response.json()
                logger.info(f"Successfully sent homework {homework.id} to Heartbeat channel {channel_id}")
                return response_data
    except aiohttp.ClientError as e:
        logger.error(f"Failed to send homework to Heartbeat: {str(e)}")
        return None 