from django.urls import path
from .views import RetrieveView, CmtSepView, RetrieveTreeView

urlpatterns = [
    path("retrieve/<str:topic>", RetrieveView.as_view()),
    path("retrieve/tree/<str:topic>", RetrieveTreeView.as_view()),
    path("retrieve/cmtsep/<str:lang>", CmtSepView.as_view()),
]