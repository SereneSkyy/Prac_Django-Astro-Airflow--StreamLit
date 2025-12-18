from rest_framework import serializers
from .models import MovieData

class IngestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MovieData
        fields = '__all__' 