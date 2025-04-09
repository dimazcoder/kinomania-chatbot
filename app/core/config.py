import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Config(BaseModel):
    env: str = os.getenv("ENV", "dev")

    project_title: str = "Kinomania Chatbot"
    project_version: str = "0.0.1"

    datetime_format: str = "%Y-%m-%dT%H:%M:%S"
    date_format: str = "%Y-%m-%d"

    cors_origins: List[str] = ['*']

    redis_host: str = os.getenv("REDIS_HOST", "")
    redis_port: str = os.getenv("REDIS_PORT", "")
    redis_password: str = os.getenv("REDIS_PASSWORD", "")

    mongo_host: str = os.getenv("MONGO_HOST", "")
    mongo_port: str = os.getenv("MONGO_PORT", "")
    mongo_password: str = os.getenv("MONGO_PASSWORD", "")
    mongo_uri: str = os.getenv("MONGO_URI", "")

    telegram_bot_token: str = os.getenv("BOT_TOKEN", "")

    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_path: str = os.path.join(project_root, 'static')
    root_dir: str = os.path.abspath(os.path.dirname(__file__))

config = Config()
