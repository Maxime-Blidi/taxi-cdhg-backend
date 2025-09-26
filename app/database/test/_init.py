import os, sys
import time
import numpy as np

MAIN_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MAIN_DIR_PATH)

from database import DatabaseElasticsearch, DatabaseRelationalPostgreSQL
from constants import LOGS_DIR, CONFIG_DIR, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER, POSTGRES_SCHEMA, \
                    POSTGRES_MASTER_PASSWORD, POSTGRES_MASTER_USER, POSTGRES_DATABASE, \
                    ELASTICSEARCH_HOST, ELASTICSEARCH_PASSWORD, ELASTICSEARCH_USER, VECTOR_DB_TYPE, \
                    AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET, AWS_REGION_NAME 

# Init the PostgreSQL DB using the master user
api_app_db = DatabaseRelationalPostgreSQL(schema=POSTGRES_SCHEMA, 
                                 host=POSTGRES_HOST,
                                 user=POSTGRES_USER,
                                 password=POSTGRES_PASSWORD,
                                 port=POSTGRES_PORT,
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_ACCESS_KEY_SECRET,
                                 aws_region_name=AWS_REGION_NAME)
# Init with master user
api_app_db._init_db(database=POSTGRES_DATABASE, 
                     schema=POSTGRES_SCHEMA,
                     user=POSTGRES_USER,
                     master_user=POSTGRES_MASTER_USER,
                     master_password=POSTGRES_MASTER_PASSWORD)
# Create with master user
api_app_db.connect_database(database=POSTGRES_DATABASE, user=POSTGRES_MASTER_USER, password=POSTGRES_MASTER_PASSWORD)
api_app_db.execute_file(file_path=os.path.join(CONFIG_DIR, "postgres_setup.sql"))
# Now interract with prod user
api_app_db.connect_database(database=POSTGRES_DATABASE, user=POSTGRES_USER, password=None)
print(api_app_db.query_data(SELECT="*",FROM="users"))

