import os, sys
import logging
from dotenv import load_dotenv
import json

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"), override=True)

logger = logging.Logger(name="Constants")

# CONSTANTS
MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(MAIN_DIR)
TMP_DIR = os.path.join(ROOT_DIR, "tmp")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
BIN_DIR = os.path.join(ROOT_DIR, "bin")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# SECRETS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_ACCESS_KEY_SECRET = os.getenv("AWS_ACCESS_KEY_SECRET", "")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "")

POSTGRES_MASTER_USER=os.getenv("POSTGRES_MASTER_USER", "")
POSTGRES_MASTER_PASSWORD=os.getenv("POSTGRES_MASTER_PASSWORD", "")
POSTGRES_DATABASE=os.getenv("POSTGRES_DATABASE", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "")
