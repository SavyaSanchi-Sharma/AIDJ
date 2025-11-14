"""
Task service for managing asynchronous processing jobs.
"""

import structlog
from typing import Dict, Any
from uuid import UUID
from celery import Celery

from ..core.config import get_settings

logger = structlog.get_logger(__name__)


class TaskService:
    """Service for managing asynchronous tasks."""
    
    def __init__(self):
        self.settings = get_settings()
        self.celery_app = Celery(
            'music-mixer',
            broker=self.settings.RABBITMQ_URL,
            backend=self.settings.REDIS_URL
        )

    async def start_song_processing(self, song_id: UUID) -> str:
        """
        Start asynchronous song processing pipeline.
        
        Returns task ID for tracking.
        """
        task = self.celery_app.send_task(
            'stem_processing.process_song',
            args=[str(song_id)],
            queue='audio_processing'
        )
        
        logger.info("Song processing task started", song_id=song_id, task_id=task.id)
        return task.id

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a processing task."""
        result = self.celery_app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "info": result.info
        }