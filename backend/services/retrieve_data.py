from django.db import connection
from django.db.utils import ProgrammingError
from rest_framework.exceptions import ValidationError
from api.retrieve.serializers import CommentsSerializer

def retrieve_data():
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM airflow.comments LIMIT 10;")
            cols = [c[0] for c in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]

        # Serialize (and validate) only the fields we care about
        serializer = CommentsSerializer(data=rows, many=True)
        serializer.is_valid(raise_exception=True)
        return serializer.data

    except ProgrammingError as e:
        return {"error": "database_error", "detail": str(e)}
    except ValidationError as e:
        return {"error": "validation_error", "detail": e.detail}
    except Exception as e:
        return {"error": "unknown_error", "detail": str(e)}
