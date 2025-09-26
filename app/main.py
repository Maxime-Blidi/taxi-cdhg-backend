
from fastapi import FastAPI
import os
import uvicorn
import logging


from constants import LOGS_DIR, CONFIG_DIR, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER, POSTGRES_SCHEMA, POSTGRES_DATABASE, \
                    POSTGRES_MASTER_USER, POSTGRES_MASTER_PASSWORD, \
                    AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_REGION_NAME
from database import DatabaseRelationalPostgreSQL

from api import include_routers_v1


# Ensures db and schemas are setup
db = DatabaseRelationalPostgreSQL(schema=POSTGRES_SCHEMA, 
                                    database=POSTGRES_DATABASE,
                                    host=POSTGRES_HOST,
                                    user=POSTGRES_USER,
                                    port=POSTGRES_PORT,
                                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                                    aws_region_name=AWS_REGION_NAME,
                                    aws_secret_access_key=AWS_ACCESS_KEY_SECRET)
# db._config_logger(logs_name="DatabaseRelationalPostgreSQL", logs_level="ERROR")
# db.drop_schema(schema_name=POSTGRES_SCHEMA)
db._init_db(database=POSTGRES_DATABASE,
            schema=POSTGRES_SCHEMA,
            user=POSTGRES_USER,
            master_user=POSTGRES_MASTER_USER,
            master_password=POSTGRES_MASTER_PASSWORD)
db.execute_file(file_path=os.path.join(CONFIG_DIR, "postgres_setup.sql"))

# Starts the server
app = FastAPI()
include_routers_v1(app=app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

