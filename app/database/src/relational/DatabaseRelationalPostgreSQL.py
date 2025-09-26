import os, sys
import psycopg2
from psycopg2 import sql, OperationalError, errors
from psycopg2._psycopg import connection
from psycopg2.extras import DictCursor, register_uuid
from psycopg2.extensions import register_type, UNICODE
from typing import Union, Optional, Any
import json
import boto3
import psycopg2._psycopg

from .DatabaseRelational import DatabaseRelational


class DatabaseRelationalPostgreSQL(DatabaseRelational):
    """
    A class to manage PostgreSQL databases (RDS, Aurora, local) with optional IAM authentication.
    """

    def __init__(self,
                 host: str = "127.0.0.1",
                 database: str = "app_database",
                 schema: str = "app_schema",
                 user: str = "app_user",
                 password: Optional[str] = None,
                 port: str = "5432",
                 ssl_mode: Optional[str] = None,
                 connection_timeout: int = 30,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region_name: Optional[str] = None,
                 ) -> None:
        """
        A high-level interface for PostgreSQL server database, compatible with standard PostgreSQL, 
        AWS Aurora PostgreSQL, and AWS RDS PostgreSQL.

        For an explanation of the 'database' and 'schema' denomination, see the Notes below.

        Parameters
        ----------
        schema: str
            The name of the schema (can be seen as the database name) to connect to. Having multiple databases on a same server is not
            supported, so the server management is limited to schema level. 
        host: str
            The host/address of the database server.
            - When connecting to a local server, the IP of host computer
            - When connecting to AWS, the address of the read/write endpoint
        user: str
            The user to assume when interacting with the DB
            - Must be an existing DB user
            - Can be an IAM user, provided an homonymous IAM user exists
        password: str
            The password for the database user (not used if IAM authentication is enabled)
            - When a password is given, will assume a direct connection to the DB using the given user and password
            - When password is None, will assume an IAM authentication, so a connection token will be generated for this IAM user
        port: str
            The port number for the database connection.
        ssl_mode: Optional[str]
            SSL mode for the connection (e.g., 'require', 'verify-ca', 'verify-full').
        connection_timeout: int
            Connection timeout in seconds.
        aws_access_key_id: Optional[str]
            AWS access key ID for IAM authentication.
        aws_secret_access_key: Optional[str]
            AWS secret access key for IAM authentication.
        aws_region_name: Optional[str]
            AWS region name for IAM authentication.

        Notes
        -----
        - The 'database' in common words often refers to a 'schema' in technical terms. Thus, a database can 
        rather be seen as a server, and a schema as a database.
            - A schema is a collection of tables
            - A database is a collection of schemas
        - Database management is a rather uncommon operation. For a more streamlined usage of this helper, once connected, \
        management is limited to schema level.
        """
        super().__init__(logs_name="DatabaseRelationalPostgreSQL")

        self.host = host
        self.database = database
        self.schema = schema.lower().replace("-", "_").replace(" ", "_")
        self.user = user
        self.password = password
        self.port = port
        self.ssl_mode = ssl_mode
        self.connection_timeout = connection_timeout
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.conn: psycopg2._psycopg.connection = None  # type: ignore

        register_uuid()
        register_type(UNICODE) # Register UUID type

        return None

    def _run(self, query, params=None):
        """Run simple admin management query."""
        query_str = str(query).replace("\n", " ").strip()
        self.logger.debug(f"Running SQL: {query_str} | Params: {params}")

        try:
            if self.conn is not None:
                with self.conn.cursor() as cur:
                    cur.execute(query, params)
                self.logger.debug(f"SQL query succeeded: {query_str}")
            else:
                self.logger.error("Could not run SQL query, connection is closed.")
        except Exception as e:
            self.logger.error(f"SQL query failed: {query_str} | Error: {e}", exc_info=True)
        
        return None


    def connect_database(self, 
                         database: str, 
                         user: str, 
                         password: Optional[str] = None):
        """
        Get a psycopg2 ``conn`` connection to a specified database.
        """

        def _get_connection_params(database: str, connect_user: str, password: Optional[str]):
            """
            Build psycopg2 connection parameters.
            If password is None → generate IAM token.
            """
            params = {
                "host": self.host,
                "user": connect_user,
                "dbname": database,
                "port": self.port,
                "connect_timeout": self.connection_timeout,
            }

            if self.ssl_mode:
                params["sslmode"] = self.ssl_mode

            # IAM auth if no password
            if password is None or password=="":
                self.logger.info(f"Using IAM auth for user '{connect_user}'")
                client = boto3.client(
                    "rds",
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region_name,
                )
                token = client.generate_db_auth_token(
                    DBHostname=self.host,
                    Port=int(self.port),
                    DBUsername=connect_user,
                    Region=self.aws_region_name,
                )
                params["password"] = token
                if "sslmode" not in params:
                    params["sslmode"] = "require"
            else:
                params["password"] = password

            return params

        # Enforce disconnection from other DB
        try:
            self.conn.close()
            self.conn = None # type: ignore
        except:
            self.conn = None # type: ignore

        try:
            conn_params = _get_connection_params(database, user, password)
            self.conn = psycopg2.connect(**conn_params)
            self.conn.autocommit = True

            self.conn.cursor().execute(f"SET search_path TO {self.schema}, public;")
            self._commit()

            self.logger.info(f"Connector created for database={database}")
            self.logger.info(f"Connector path set to: '{self.schema}'.")

            return None
        
        except Exception as e:
            self.logger.critical(f"Database connection failed: {e}")
            return None


    def _init_db(self,
                database: str = "app_database",
                schema: str = "app_schema",
                user: str = "app_user",
                master_user: str = "postgres",
                master_password: Optional[str] = "password"):
        """
        Initialize the target database, schema, and user with automatic IAM vs password detection.
        - If master_password is None → IAM token authentication is used.
        """

        schema = schema.lower().replace("-", "_").replace(" ", "_")
        iam_mode = True if self.aws_access_key_id is not None and self.aws_region_name is not None and self.aws_secret_access_key is not None else False
        self.logger.info(
            f"Initializing DB '{database}', schema '{schema}', user '{user}' "
            f"using {'IAM' if iam_mode else 'password'} auth"
        )

        # 1. Connect to 'postgres' as master
        self.connect_database("postgres", master_user, master_password)

        # 2. Create target DB if missing
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("app_database")))
            self.conn.close()
        except Exception as e:
            if "already exists" in str(e):
                self.logger.warning(f"Database '{database}' already exists and won't be recreated unless dropped.")
            else:
                self.logger.error(str(e))

        # 3. Connect to target DB
        self.connect_database(database, master_user, master_password)

        # 4. Create app user
        self._run(sql.SQL("DO $$ BEGIN CREATE ROLE {} LOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;")
                .format(sql.Identifier(user)))
        if iam_mode:
            self._run(sql.SQL("GRANT rds_iam TO {};").format(sql.Identifier(user)))

        # 5. Lock down public schema
        self._run("REVOKE CREATE ON SCHEMA public FROM PUBLIC;")
        self._run(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC;").format(sql.Identifier(database)))

        # 6. Create schema safely
        self._run(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema)))
        self._run(sql.SQL("ALTER SCHEMA {} OWNER TO {};").format(sql.Identifier(schema), sql.Identifier(user)))

        # 7. Grant privileges
        self._run(sql.SQL("GRANT CONNECT ON DATABASE {} TO {};").format(sql.Identifier(database), sql.Identifier(user)))
        self._run(sql.SQL("GRANT USAGE ON SCHEMA {} TO {};").format(sql.Identifier(schema), sql.Identifier(user)))
        self._run(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {};")
                .format(sql.Identifier(schema), sql.Identifier(user)))
        self._run(sql.SQL("GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA {} TO {};")
                .format(sql.Identifier(schema), sql.Identifier(user)))

        # 8. Default privileges for future objects
        self._run(sql.SQL("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA {}
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {};
        """).format(sql.Identifier(schema), sql.Identifier(user)))
        self._run(sql.SQL("""
            ALTER DEFAULT PRIVILEGES IN SCHEMA {}
            GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {};
        """).format(sql.Identifier(schema), sql.Identifier(user)))

        # 9. Set search_path
        self._run(sql.SQL("SET search_path TO {}, public;").format(sql.Identifier(schema)))
        self.logger.info(f"DB setup complete: Database={database}, Schema={schema}, User={user} (IAM={iam_mode})")

        return None
        

    def create_table(self, table_name: str, column_definitions: list[str]):
        """
        Creates a table in the PostgreSQL database.

        Parameters
        ----------
        table_name: str
            Name of the table to create
        column_definitions: list[str]
            List of column definitions (e.g., ["id SERIAL PRIMARY KEY", "name VARCHAR(100) NOT NULL"])

        Returns
        -------
        bool
            True if table was created or already exists and is accessible, False otherwise
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            with self.conn.cursor() as cursor:
                full_table_name = f"{self.schema}.{table_name}"

                create_table_sql = f"CREATE TABLE IF NOT EXISTS {full_table_name} ({', '.join(column_definitions)})"
                cursor.execute(create_table_sql)

                cursor.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)", 
                            (self.schema, table_name))
                table_exists = cursor.fetchone()[0] # type: ignore

                if table_exists:
                    self.logger.info(f"Table '{self.schema}.{table_name}' successfully created or already exists.")
                    self._commit()
                else:
                    self.logger.warning(f"Failed to create table '{self.schema}.{table_name}'.")
                    self._rollback()

        except Exception as e:
            self._rollback()
            self.logger.error(f"Error creating table '{table_name}': {e}")

        return None


    def delete_data(self,
                    FROM: str,
                    WHERE: Union[str, list[str], tuple[str]],
                    VALUES: Optional[Union[str, int, list, tuple]] = None,
                    LIKE: Optional[Union[str, list[str], tuple[str]]] = None):
        """
        Removes data from a PostgreSQL table under optional conditions.

        Parameters
        ----------
        FROM: str
            The table name to delete from.
        WHERE: str, list[str]
            The column(s) to use in the WHERE clause.
        VALUES: Optional[str, int, list, tuple]
            Values to use for exact matches.
        LIKE: Optional[str, list[str]]
            Pattern(s) to use for LIKE matching.

        Notes
        -----
        - Cascading is handled by the database schema via ``ON DELETE CASCADE`` constraints.
        """
        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor()

            if not WHERE:
                self.logger.warning("Deleting the whole data without WHERE clause is not suppported. Consider using a joker LIKE query.")
                return None

            where_cols = [WHERE] if isinstance(WHERE, str) else list(WHERE)

            # Exact match
            if VALUES is not None:
                values = [VALUES] if not isinstance(VALUES, (list, tuple)) else list(VALUES)
                if len(where_cols) != len(values):
                    self.logger.warning("Number of WHERE columns must match number of VALUES.")
                    return None

                where_clause = " AND ".join([f"{col} = %s" for col in where_cols])
                query = f"DELETE FROM {FROM} WHERE {where_clause};"
                cursor.execute(query, tuple(values))
                self.logger.debug(f"{query} -- {values}")

            # LIKE match
            elif LIKE is not None:
                patterns = [LIKE] if not isinstance(LIKE, (list, tuple)) else list(LIKE)
                if len(where_cols) != len(patterns):
                    self.logger.warning("Number of WHERE columns must match number of LIKE patterns.")
                    return None

                where_clause = " AND ".join([f"{col} LIKE %s" for col in where_cols])
                query = f"DELETE FROM {FROM} WHERE {where_clause};"
                cursor.execute(query, tuple(patterns))
                self.logger.debug(f"{query} -- {patterns}")

            self._commit()

        except Exception as e:
            self.logger.error(f"PostgreSQL error when deleting data: {e}")

        finally:
            cursor.close()

        return None



    def disconnect_database(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            self.logger.info(f"Disconnected from database schema '{self.schema}'.")

        return None


    def drop_table(self, table_name: str):
        """
        Drops a table from the database.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor()
            drop_table_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
            cursor.execute(drop_table_query)
            self._commit()
            self.logger.info(f"Successfully dropped table `{table_name}`.")
            cursor.close()
        except Exception as e:
            self.logger.error(f"Failed to drop table `{table_name}`: {e}")
            self._rollback()

        return None


    def drop_schema(self, schema: str):
        """
        Drops a schema from the PostgreSQL database.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )
            self._commit()
            cursor.close()
            self.logger.info(f"Successfully dropped schema '{schema}'.")

        except Exception as e:
            self.logger.error(f"PostgreSQL error when dropping schema: {e}")
            self._rollback()

        return None


    def drop_database(self, schema: str):
        """
        See ``drop_schema()``.
        """
        self.logger.debug("Drop database not allowed. Drop schemas instead.")
        return self.drop_schema(schema=schema)


    def list_databases(self, display: bool = False):
        """
        Lists all databases on the server.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:

            cursor = self.conn.cursor()
            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
            databases_list = [db[0] for db in cursor.fetchall()]

            self.logger.debug("Available databases:", databases_list)
            if display:
                print("Available databases:", databases_list)
            
            cursor.close()
            return databases_list
        
        except Exception as e:
            self.logger.error(f"PostgreSQL error when listing databases: {e}")
            return []


    def list_schemas(self, include_system_schemas: bool = False, display: bool = False):
        """
        Lists all schemas in the specified database or in the currently connected database.

        Parameters
        ----------
        include_system_schemas: bool, default=False
            Whether to include system schemas (pg_*, information_schema) in the results.
        display: bool, default=False
            Whether to print the list of schemas.

        Returns
        -------
        list
            A list of schema names.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor()

            if include_system_schemas:
                query = "SELECT schema_name FROM information_schema.schemata ORDER BY schema_name;"
            else:
                query = """
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT LIKE 'pg_%' 
                AND schema_name != 'information_schema'
                ORDER BY schema_name;
                """

            cursor.execute(query)
            schemas_list = [schema[0] for schema in cursor.fetchall()]

            self.logger.debug(f"Schemas in database:", schemas_list)
            if display:
                print(f"Schemas in database:", schemas_list)

            cursor.close()
            return schemas_list

        except Exception as e:
            self.logger.info(f"PostgreSQL error when listing schemas: {e}")
            return []
                

    def list_tables(self, display: bool = False):
        """
        Lists all tables in the specified PostgreSQL schema or the current schema.

        Parameters
        ----------
        display: bool, default=False
            Whether to print the list of tables.

        Returns
        -------
        list
            A list of table names.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:

            cursor = self.conn.cursor()

            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.schemata WHERE schema_name = %s
                );
            """, (self.schema,))
            schema_exists = cursor.fetchone()[0] # type: ignore

            if not schema_exists:
                self.logger.info(f"Schema '{self.schema}' does not exist.")
                return []

            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                ORDER BY table_name;
            """, (self.schema,))

            tables_list = [row[0] for row in cursor.fetchall()]

            self.logger.debug(f"Tables in '{self.schema}': {', '.join(tables_list)}")
            if display:
                if tables_list:
                    print(f"Tables in '{self.schema}': {', '.join(tables_list)}")
                else:
                    print(f"No tables found in schema '{self.schema}'.")

            cursor.close()
            return tables_list

        except Exception as e:
            self.logger.error(f"PostgreSQL error when listing tables: {e}")
            return []


    def query_data(self,
                SELECT: str,
                FROM: str,
                JOIN: Optional[Union[str, list[str], tuple[str]]] = None,
                WHERE: Optional[Union[str, list[str], tuple[str]]] = None,
                VALUES: Optional[Union[Any, list[Any], tuple[Any]]] = None,
                LIKE: Optional[Union[str, list[str], tuple[Any]]] = None):
        """
        Selects data from a PostgreSQL table with optional JOIN and filtering.

        Parameters
        ----------
        SELECT : str
            The columns to select in SQL syntax.
        FROM : str
            The name of the table to select data from.
        JOIN : str or list, optional
            One or more JOIN clauses (e.g. "JOIN table2 t2 ON t1.id = t2.id").
        WHERE : str or list, optional
            Column name(s) for WHERE condition.
        VALUES : Any or list, optional
            Value(s) for exact matching in WHERE.
        LIKE : str or list, optional
            Pattern(s) for LIKE matching in WHERE.

        Returns
        -------
        rows : list[dict]
            Rows retrieved as a list of dicts.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor(cursor_factory=DictCursor)

            sql_parts = [f"SELECT {SELECT}", f"FROM {FROM}"]

            if JOIN is not None:
                joins = [f"JOIN {JOIN}"] if isinstance(JOIN, str) else list(f"JOIN {JOIN}")
                sql_parts.extend(joins)

            where_clause = ""
            params = []

            # If exact value query
            if VALUES is not None:
                where_cols = [WHERE] if isinstance(WHERE, str) else WHERE
                values = [VALUES] if not isinstance(VALUES, (list, tuple)) else VALUES
                if len(where_cols) != len(values): # type: ignore
                    self.logger.warning("Number of WHERE columns must match number of VALUES.")
                    cursor.close()
                    return []
                where_clause = " AND ".join([f"{col} = %s" for col in where_cols]) # type: ignore
                params = values

            # If like value query
            elif LIKE is not None:
                where_cols = [WHERE] if isinstance(WHERE, str) else WHERE
                like_patterns = [LIKE] if not isinstance(LIKE, (list, tuple)) else LIKE
                if len(where_cols) != len(like_patterns): # type: ignore
                    self.logger.warning("Number of WHERE columns must match number of LIKE patterns.")
                    cursor.close()
                    return []
                where_clause = " AND ".join([f"{col} LIKE %s" for col in where_cols]) # type: ignore
                params = like_patterns

            if where_clause:
                sql_parts.append(f"WHERE {where_clause}")

            sql_query = " ".join(sql_parts) + ";"
            self.logger.debug(sql_query)
            cursor.execute(sql_query, params)
            rows = [dict(row) for row in cursor.fetchall()]

            cursor.close()
            return rows

        except psycopg2.Error as e:
            self.logger.error(f"PostgreSQL error when selecting data: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when selecting data: {e}")
            return []


    def send_data(self, table_name: str, **kwargs):
        """
        Inserts data into a PostgreSQL table. Infers column-value pairs from ``kwargs`` key-value pairs.

        Parameters
        ----------
        table_name: str
            The name of the table to insert into.
        **kwargs
            Column-value pairs to update.

        Examples
        --------
        >>> send_data(table_name="users", user_id=42, user_name="jdoe")
        """
        
        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor()
            fields = ",".join(list(kwargs.keys()))
            placeholders = ",".join(["%s"] * len(kwargs))
            values = tuple(kwargs.values())

            cursor.execute(f"INSERT INTO {table_name} ({fields}) VALUES ({placeholders});", values)
            self.logger.debug(f"INSERT INTO {table_name} ({fields}) VALUES ({values});")
            self._commit()

        except AttributeError:
            self.logger.error(f"Connection to the DB is not defined. Try reconnecting to the DB.")

        except Exception as e:
            self.logger.error(f"PostgreSQL error when inserting data into '{table_name}': {e}")

        finally:
            self._rollback()

        return None


    def update_data(self,
                    table_name: str,
                    WHERE: Optional[Union[str, list[str], tuple[str]]] = None,
                    VALUES: Optional[Union[Any, list[Any], tuple[Any]]] = None,
                    LIKE: Optional[Union[str, list[str], tuple[str]]] = None,
                    **kwargs):
        """
        Updates data in a PostgreSQL table based on a WHERE clause.

        Parameters
        ----------
        table_name: str
            The name of the table to update.
        WHERE: Optional[str or list[str]]
            The column(s) to use in the WHERE clause.
        VALUES: Optional[Any or list[Any]]
            The values to use for exact matches.
        LIKE: Optional[str or list[str]]
            The pattern(s) to use for LIKE matching.
        **kwargs
            Column-value pairs to update.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            cursor = self.conn.cursor()

            # Prepare SET clause
            set_clause = ", ".join([f"{field} = %s" for field in kwargs])
            set_values = tuple(kwargs.values())

            # Prepare WHERE clause
            where_clause = ""
            where_values = ()

            if WHERE is not None:
                where_cols = [WHERE] if isinstance(WHERE, str) else list(WHERE)

                if VALUES is not None:
                    values = [VALUES] if not isinstance(VALUES, (list, tuple)) else list(VALUES)
                    if len(where_cols) != len(values):
                        self.logger.warning("Number of WHERE columns must match number of VALUES.")
                        return None
                    where_clause = " AND ".join([f"{col} = %s" for col in where_cols])
                    where_values = tuple(values)

                elif LIKE is not None:
                    patterns = [LIKE] if not isinstance(LIKE, (list, tuple)) else list(LIKE)
                    if len(where_cols) != len(patterns):
                        self.logger.warning("Number of WHERE columns must match number of LIKE patterns.")
                        return None
                    where_clause = " AND ".join([f"{col} LIKE %s" for col in where_cols])
                    where_values = tuple(patterns)

            # Build final query
            query = f"UPDATE {table_name} SET {set_clause}"
            if where_clause:
                query += f" WHERE {where_clause}"
            query += ";"

            full_values = set_values + where_values
            cursor.execute(query, full_values)

            self.logger.debug(f"{query} -- {full_values}")
            self._commit()

        except AttributeError:
            self.logger.error("Connection to the DB is not defined. Try reconnecting to the DB.")

        except Exception as e:
            self.logger.error(f"PostgreSQL error when updating data in '{table_name}': {e}")

        finally:
            self._rollback()

        return None
    

    def raw_sql(self, SQL: str, VALUES: tuple[str]):

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        try:
            self.logger.warning(f"Running raw SQL query: {SQL}")
            cursor = self.conn.cursor(cursor_factory=DictCursor)
            cursor.execute(SQL, VALUES)
            rows = cursor.fetchall()
            cursor.close()
        except Exception as e:
            self.logger.critical(f"SQL raw query failed:\n {e}")

        return rows


    def _commit(self):
        """
        Handles the commits of the transactions operated since last commit.
        """
        try:
            self.conn.commit()
        except AttributeError:
            self.logger.error("Connector is not defined. Try reconnecting to the DB.")
        except:
            self.logger.error("Failed to commit transaction.")

        return None
    

    def _rollback(self):
        """
        Handles the roolbacks of the transactions operated since last commit.
        """
        try:
            self.conn.rollback()
        except AttributeError:
            self.logger.error("Connector is not defined. Try reconnecting to the DB.")
        except:
            self.logger.error("Failed to rollback transaction.")

        return None


    def execute_file(self, file_path: str):
        """
        Executes SQL commands from a .sql file or inserts data from a .json file.

        Parameters
        ----------
        file_path : str
            The path to the .sql or .json file.
        """

        if self.conn is None:
            self.connect_database(database=self.database, user=self.user, password=self.password)

        if not os.path.isfile(file_path):
            self.logger.warning(f"File '{file_path}' does not exist.")
            return None


        try:
            cursor = self.conn.cursor()

            file_extension = os.path.splitext(file_path)[1]
            if file_extension == '.sql':
                with open(file_path, 'r', encoding='utf-8') as sql_file:
                    sql_commands = sql_file.read()
                    cursor.execute(sql_commands)
                    self.conn.commit()
                    self.logger.info(f"SQL file '{file_path}' executed successfully.")

            elif file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                if not isinstance(data, list):
                    self.logger.warning("JSON file must contain a list of records (dictionaries).")
                    return None

                for record in data:
                    if not isinstance(record, dict) or 'table' not in record or 'data' not in record:
                        self.logger.warning("JSON format invalid. Each record must contain 'table' and 'data' keys.")
                        return None

                    table_name = record['table']
                    fields = ', '.join(record['data'].keys())
                    placeholders = ', '.join(['%s'] * len(record['data']))
                    values = tuple(record['data'].values())

                    cursor.execute(f"INSERT INTO {table_name} ({fields}) VALUES ({placeholders});", values)

                self.conn.commit()
                self.logger.info(f"JSON file '{file_path}' data inserted successfully.")

            else:
                self.logger.warning(f"Unsupported file type '{file_extension}'. Only .sql and .json are supported.")

        except psycopg2.Error as e:
            self.logger.error(f"PostgreSQL error when executing file: {e}")
            self._rollback()

        return None
