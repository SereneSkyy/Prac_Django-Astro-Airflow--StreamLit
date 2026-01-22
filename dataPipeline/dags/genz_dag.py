from airflow.decorators import dag, task
import pendulum
from pendulum import datetime
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from schemas.etl_schema import (execute_comments_sql, execute_topic_sql, execute_processed_vidIds_sql, 
                                insert_comments_sql, insert_topic_sql, insert_processed_vidIds_sql,
                                update_topic_sql,execute_processed_comments_sql,execute_taxonomy_sql
                                )
from collectors.youtube_collector import YouTubeNepal
from collectors.cmt_sep_collector import cmt_sep_collector
from airflow.exceptions import AirflowSkipException
from services.psql_conn import psql_cursor
from services.redis_client import get_redis
from services.api_services import api_provider
from dataPipeline.services.run_embed import create_embeddings

# --------------------- Comments Fetching Dag ----------------------------------
@dag(
    dag_id="genz_dag",
    start_date=datetime(2023, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1},
    tags=["nepal", "genz"],
)

def start_genz_dag():
    @task
    def extract_data(**context):
        extract_info = {
            'dag_id': '',
            'topic': '',
            'max_results': 1,
            'cmt_per_vid': 15,
        }
        redis = get_redis()
        ctx = get_current_context()
        conf = (ctx.get('dag_run').conf or {})
        extract_info['topic'], extract_info['dag_id']  = conf.get('topic', 'genz'), conf.get('dag_id', 'genz_dag') 
        
        api_key = api_provider()
        if not api_key:
            reason = "No API key found, cannot fetch data"
            context['ti'].xcom_push(key='skip_reason', value=reason)
            raise AirflowSkipException(reason)

        collector = YouTubeNepal(api_key)
            
        # search videos
        vidIds = collector.search_videos(extract_info['topic'], extract_info['max_results'])
        print(f"Searching for topic: {extract_info['topic']}")

        if not vidIds:
            print("No videos found.")
            return {'items': []}

        all_items = []
        for vidId in vidIds:
            if collector.is_already_processed(vidId):
                redis.sadd(f"processed:{extract_info['dag_id']}", vidId)
                print(f"Skipping {vidId}: Already processed")
                continue

            # api call
            print(f"Data Fetching for: {vidId}")
            # not processed scenario
            redis.sadd(f"not_processed:{extract_info['dag_id']}", vidId)
            comments = collector.fetch_data(vidId, cmt_per_vid = extract_info['cmt_per_vid'])
            if comments:
                all_items.extend([
                    {
                        "vid_id": vidId,
                        "item": item,
                    }
                    for item in comments
                ])
                # collector.mark_as_processed(vidId)
                print(f"Logged ID {vidId} to processed_log.json")

            else:
                print(f"No data retrieved for {vidId}")
    
        if not all_items:
            reason = "No comments found!"
            context['ti'].xcom_push(key='skip_reason', value=reason)
            raise AirflowSkipException(reason)
            
        return {'items': all_items}

    @task
    def transform_data(extracted_data):
        items = extracted_data.get("items", [])

        comments = []
        for wrapped in items:
            # get parent video id
            vid_id = wrapped["vid_id"]          
            # actual YT comment object
            item = wrapped["item"]              

            snippet = (
                item.get("snippet", {})
                    .get("topLevelComment", {})
                    .get("snippet")
            )
            if not snippet:
                continue

            comments.append(
                {
                    # comment id
                    "id": item["id"],           
                    # keep association
                    "vid_id": vid_id,            
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
        conf = ctx.get("dag_run").conf or {}

        topic = conf.get("topic", "genz")
        dag_id = ctx["dag"].dag_id
        collector = "YT"

        redis = get_redis()

        # read redis state
        processed_vids = redis.smembers(f"processed:{dag_id}")
        not_processed_vids = redis.smembers(f"not_processed:{dag_id}")

        with psql_cursor() as cursor:
            # ensure tables exist
            cursor.execute(execute_comments_sql)
            cursor.execute(execute_topic_sql)
            cursor.execute(execute_processed_comments_sql)
            cursor.execute(execute_taxonomy_sql)
            # insert comments
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

            cursor.executemany(insert_comments_sql, comment_values)
            try:
                cmt_vals = {}
                for val in comment_values: cmt_vals[val[0]] = val[1] # -> dict of key (id): value (comment)
                cmt_sep_collector(cursor, cmt_vals)
            except Exception as e:
                print(f"Exception in cmt_sep_collector:- {e}")
            finally:
                # new videos → INSERT
                new_topic_values = [
                    (
                        val["id"],
                        [topic],
                        [collector],
                        dag_id,
                    )
                    for val in comments
                    if val["vid_id"] in not_processed_vids
                ]

                cursor.executemany(insert_topic_sql, new_topic_values)

                # processed videos → UPDATE (add topic)
                for val in comments:
                    if val["vid_id"] in processed_vids:
                        cursor.execute(
                            update_topic_sql,
                            ([topic], val["id"])
                        )

                processed_values = [
                    (
                        val["vid_id"],
                        val["id"],
                    )
                    for val in comments
                    if (
                        val["vid_id"] in processed_vids
                        or val["vid_id"] in not_processed_vids
                    )
                ]

                cursor.executemany(insert_processed_vidIds_sql, processed_values)

        redis.delete(f"processed:{dag_id}")
        redis.delete(f"not_processed:{dag_id}")
        return {"topic": topic, "source_dag_id": dag_id, "vid_ids": not_processed_vids}

    data = extract_data()
    comments = transform_data(data)
    payload = load_data(comments)

    # load the vids not processed to embed_dag
    trigger = TriggerDagRunOperator(
            task_id="trigger_embed_dag",
            trigger_dag_id="embed_dag",
            wait_for_completion=False,
            conf=payload,
        )
    payload >> trigger

# --------------------- Comments Fetching Dag ----------------------------------
@dag(
    dag_id="embed_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
)
def embed_dag():
    @task
    def bert_embed():
        ctx = get_current_context()
        conf = ctx.get("dag_run").conf or {}

        vid_ids = conf.get("vid_ids", [])
        topic = conf.get("topic", "genz")

        if not vid_ids:
            return
        
        with psql_cursor() as cursor:
            create_embeddings(vid_ids, cursor)
    
    bert_embed()

# call the dag
start_genz_dag()
embed_dag()