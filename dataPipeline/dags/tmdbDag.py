from airflow.decorators import dag, task
from airflow.hooks.base import BaseHook
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from pendulum import datetime
from schemas.etl_schema import execute_sql, insert_sql

POSTGRES_CONN_ID = "postgres_default"

@dag(
    start_date=datetime(2024, 1, 1),  # must be in the past to run
    schedule="@daily",
    catchup=False,
    default_args={"owner": "Astro", "retries": 3},
    tags=["tmdb"],
)
def start_tmdb():

    @task
    def extract_tmdb_data():
        conn = BaseHook.get_connection("tmdb_api")
        tmdb_api_key = conn.password

        http_hook = HttpHook(http_conn_id="tmdb_api", method="GET")
        endpoint = f"/3/movie/popular?api_key={tmdb_api_key}&page=1"

        response = http_hook.run(endpoint)

        if response.status_code == 200:
            return response.json()

        raise Exception(f"Failed to fetch TMDB data: {response.status_code}")

    @task
    def transform_tmdb_data(tmdb_data):
        movies = tmdb_data["results"]
        transformed_movies = []

        for movie in movies:
            transformed_movies.append(
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "popularity": movie["popularity"],
                    "release_date": movie["release_date"],
                    "vote_count": movie["vote_count"],
                }
            )

        return transformed_movies

    @task
    def load_tmdb_data(transformed_data):
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        cursor.execute(execute_sql)

        values = [
            (
                m["id"],
                m["title"],
                m["popularity"],
                m["release_date"],
                m["vote_count"],
            )
            for m in transformed_data
        ]

        cursor.executemany(insert_sql, values)
        conn.commit()
        cursor.close()

    tmdb_data = extract_tmdb_data()
    transformed_data = transform_tmdb_data(tmdb_data)
    load_tmdb_data(transformed_data)


start_tmdb()
