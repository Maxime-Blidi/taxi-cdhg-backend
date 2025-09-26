
from fastapi import FastAPI
import os
import uvicorn
import logging


from constants import LOGS_DIR, CONFIG_DIR, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER, POSTGRES_SCHEMA, POSTGRES_DATABASE, \
                    POSTGRES_MASTER_USER, POSTGRES_MASTER_PASSWORD, \
                    AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_REGION_NAME
from database import DatabaseRelationalPostgreSQL

from api import include_routers_v1


# Starts the server
app = FastAPI()
include_routers_v1(app=app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

