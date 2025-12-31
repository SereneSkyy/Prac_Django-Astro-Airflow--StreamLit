from rest_framework.views import APIView
from rest_framework.response import Response
from services.retrieve_data import retrieve_data, cmt_sep_data

class RetrieveView(APIView):
    def get(self, request, topic):        
        result = retrieve_data(topic)
        return Response(result)

class CmtSepView(APIView):
    def get(self, request, lang):        
        result = cmt_sep_data(lang)
        return Response(result)
