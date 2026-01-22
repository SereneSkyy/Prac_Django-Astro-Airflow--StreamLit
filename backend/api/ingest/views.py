from rest_framework.views import APIView
from services.airflow import trigger_dag
from services.exceptions import AirflowError
from django.http import JsonResponse

class IngestView(APIView):
    def post(self, request, topic: str = 'genz', dag_id : str = 'genz_dag'):
        try:
            result = trigger_dag(dag_id, topic)
            return JsonResponse(result, status=200)

        except AirflowError as e:
            return JsonResponse({"error": str(e)}, status=502)

        except Exception as e:
            return JsonResponse({"error": "Internal server error"}, status=500)
