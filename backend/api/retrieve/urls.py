from django.urls import path
from .views import RetrieveView, CmtSepView

urlpatterns = [
    path("retrieve/<str:topic>", RetrieveView.as_view()),
    path("retrieve/cmtsep/<str:lang>", CmtSepView.as_view()),
]