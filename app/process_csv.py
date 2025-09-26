
import sys, os
import pandas as pd


from constants import LOGS_DIR, CONFIG_DIR, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER, POSTGRES_SCHEMA, POSTGRES_DATABASE, \
                    POSTGRES_MASTER_USER, POSTGRES_MASTER_PASSWORD, \
                    AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_REGION_NAME
from database import DatabaseRelationalPostgreSQL


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