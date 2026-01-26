import json
from django.db import connection
from api.retrieve.serializers import CommentsSerializer

def _execute_and_serialize(sql: str, parms: list, serializer_cls, serialize_mode):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, parms)
            cols = [col[0] for col in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]

        if not rows:
            return []

        serializer = serializer_cls(data=rows, many=True, mode=serialize_mode)
        # Check validation and print to terminal if it fails
        if not serializer.is_valid():
            print(f"!!! SERIALIZER ERRORS: {serializer.errors}")
            return [{"error": "validation_error", "detail": str(serializer.errors)}]
            
        return serializer.data

    except Exception as e:
        print(f"!!! SYSTEM ERROR: {str(e)}")
        return [{"error": "system_error", "detail": str(e)}]

def retrieve_data(topic: str):
    """
    Fetches comments for the topic
    """
    sql = """
        SELECT DISTINCT c.id, c.comment, c.author, c.p_timestamp, c.t_timestamp
        FROM airflow.comments c
        JOIN airflow.topic_collector tc ON tc.id = c.id
        WHERE %s = ANY(tc.topic);
    """

    return _execute_and_serialize(sql, [topic], CommentsSerializer, 'preview')

def cmt_sep_data(lang: str):
    """
    Fetches all comments filtered by a specific language (e.g., 'en').
    """
    sql = """
        SELECT 
            c.id, 
            c.comment, 
            c.author, 
            c.p_timestamp, 
            c.t_timestamp,
            cl.language
        FROM airflow.comments c
        INNER JOIN airflow.comment_lang cl ON c.id = cl.comment_id
        WHERE cl.language = %s;
    """
    return _execute_and_serialize(sql, [lang], CommentsSerializer, 'sep')