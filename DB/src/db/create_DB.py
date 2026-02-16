import os
import psycopg
import logging

conn_params = f"host=localhost user=postgres password={os.getenv('POSTGRES_PASSWORD')}"


def createDB():
    try:
        with psycopg.connect(conn_params, autocommit=True) as conn:
            with conn.cursor() as cur:
                new_db_name = 'web_RAG_DB'
                cur.execute(f"CREATE DATABASE {new_db_name}")
                logging.info(f"Created new database {new_db_name}")
    except Exception as e:
        logging.error(e)










