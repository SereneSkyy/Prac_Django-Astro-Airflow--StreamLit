from airflow.decorators import dag, task
import pendulum
from pendulum import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import get_current_context
from schemas.etl_schema import execute_comments_sql, execute_topic_sql, insert_comments_sql, insert_topic_sql
from collectors.youtube_collector import YouTubeNepal
from airflow.exceptions import AirflowSkipException

POSTGRES_CONN_ID = "postgres_default"

@dag(
    dag_id="genz_dag",
    start_date=datetime(2023, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1},
    tags=["nepal", "genz"],
)

def start_genz_dag():
    dag_run_info = {
        'topic': '',
        'max_results': 5,
        'cmt_per_vid': 5,
    }
    @task
    def extract_data():
        ctx = get_current_context()
        conf = (ctx.get('dag_run').conf or {})
        dag_run_info['topic'] = conf.get('topic', 'genz') 
        
        collector = YouTubeNepal()

        # search videos
        vidIds = collector.search_videos(dag_run_info['topic'], dag_run_info['max_results'])
        print(f'Searching for topic: {dag_run_info['topic']}')

        if not vidIds:
            print("No videos found.")
            return {'items': []}

        all_items = []
        for vidId in vidIds:
            if collector.is_already_processed(vidId):
                print(f"Skipping {vidId}: Already processed")
                continue

            # api call
            print(f"Data Fetching for: {vidId}")
            response = collector.fetch_data(vidId, cmt_per_vid = dag_run_info['cmt_per_vid'])
            if response:
                all_items.extend(response.get("items"))
                # collector.mark_as_processed(vidId)
                print(f"Logged ID {vidId} to processed_log.json")

            else:
                print(f"No data retrieved for {vidId}")

        return {'items': all_items}

    @task
    def transform_data(extracted_data):
        items = extracted_data.get("items", [])

        if not items:
            raise AirflowSkipException("No comments found")
        comments = []
        for item in items:
            snippet = (
                item.get("snippet", {})
                    .get("topLevelComment", {})
                    .get("snippet")
            )
            if not snippet:
                continue
            comments.append(
                {
                    "id": item["id"],
                    "comment": snippet["textDisplay"],
                    "author": snippet["authorDisplayName"],
                    "p_timestamp": snippet["publishedAt"],
                    "t_timestamp": pendulum.now("Asia/Kathmandu")
                }
            )
        if not comments:
            raise AirflowSkipException("No valid comments after filtering")

        return comments

    @task
    def load_data(comments):
        ctx = get_current_context()
        conf = (ctx.get('dag_run').conf or {})
        topic, dag_id = conf.get('topic', 'genz'), conf.get('dag_id', 'genz_dag')
        collector = 'YT'

        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        cursor = conn.cursor()  

        cursor.execute(execute_comments_sql)
        cursor.execute(execute_topic_sql)

        comment_values = [
            (
                val["id"],
                val["comment"],
                val["author"],
                val["p_timestamp"],
                val["t_timestamp"],
            )
            for val in comments
        ]

        topic_values = [
            (
                val["id"],
                [topic],
                [collector],
                dag_id,
            )
            for val in comments
        ]

        cursor.executemany(insert_comments_sql, comment_values)
        cursor.executemany(insert_topic_sql, topic_values)
        conn.commit()
        cursor.close()

    data = extract_data()
    comments = transform_data(data)
    load_data(comments)

# call the dag
start_genz_dag()
