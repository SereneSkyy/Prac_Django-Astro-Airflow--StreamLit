from contextlib import contextmanager
from airflow.providers.postgres.hooks.postgres import PostgresHook
from pgvector.psycopg2 import register_vector

POSTGRES_CONN_ID = "postgres_default"

@contextmanager
def psql_cursor():
    conn = None
    cursor = None
    # to send vec vals, conn needs to be registered
    try:
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        register_vector(conn)
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
