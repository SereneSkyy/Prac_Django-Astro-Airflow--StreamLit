from rest_framework.views import APIView
from rest_framework.response import Response
from services.airflow import trigger_dag

class IngestView(APIView):
    def post(self, request):
        result = trigger_dag("start_tmdb")
        return Response(result)