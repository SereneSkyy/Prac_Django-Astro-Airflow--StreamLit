from django.db import connection
from django.db.utils import ProgrammingError
from rest_framework.exceptions import ValidationError
from api.retrieve.serializers import CommentsSerializer

def _execute_and_serialize(sql: str, params: list, serializer_cls):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            cols = [col[0] for col in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]

        serializer = serializer_cls(data=rows, many=True)
        serializer.is_valid(raise_exception=True)
        return serializer.data

    except ProgrammingError as e:
        return [{
            "error": "database_error",
            "detail": str(e),
        }]

    except ValidationError as e:
        return [{
            "error": "validation_error",
            "detail": e.detail,
        }]

    except Exception as e:
        return [{
            "error": "unknown_error",
            "detail": str(e),
        }]

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
            pc.language,
            pc.cleaned_text
        FROM airflow.comments c
        INNER JOIN airflow.processed_comments pc ON c.id = pc.comment_id
        WHERE pc.language = %s;
    """
    return _execute_and_serialize(sql, [lang], CommentsSerializer)