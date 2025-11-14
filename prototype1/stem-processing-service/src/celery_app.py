"""
Celery application configuration for audio processing tasks.
"""

import os
import structlog
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Create Celery app
app = Celery('stem-processing-service')

# Configuration
app.conf.update(
    broker_url=os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'stem_processing.*': {'queue': 'audio_processing'},
    },
    worker_concurrency=2,  # Limit concurrency for GPU memory
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Import tasks
from .tasks import audio_processing  # noqa