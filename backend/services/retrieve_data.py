import json
from django.db import connection
from rest_framework.exceptions import ValidationError
from api.retrieve.serializers import CommentsSerializer

def _execute_and_serialize(sql: str, params: list, serializer_cls):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            cols = [col[0] for col in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]

        if not rows:
            return []

        serializer = serializer_cls(data=rows, many=True)
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
    Fetches comments for a topic and joins with NLP processed data.
    """
    sql = """
        SELECT 
            c.id, 
            c.comment, 
            c.author, 
            c.p_timestamp, 
            c.t_timestamp,
            pc.language,
            pc.cleaned_text
        FROM airflow.comments c
        LEFT JOIN airflow.processed_comments pc ON c.id = pc.comment_id
        WHERE c.id IN (
            SELECT tc.id
            FROM airflow.topic_collector tc
            WHERE %s = ANY(tc.topic)
        );
    """
    return _execute_and_serialize(sql, [topic], CommentsSerializer)

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
            pc.language
        FROM airflow.comments c
        INNER JOIN airflow.processed_comments pc ON c.id = pc.comment_id
        WHERE pc.language = %s;
    """
    return _execute_and_serialize(sql, [lang], CommentsSerializer)