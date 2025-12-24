from rest_framework import serializers

class CommentsSerializer(serializers.Serializer):
    id = serializers.CharField()
    comment = serializers.CharField()
    author = serializers.CharField()
    p_timestamp = serializers.DateTimeField()
    t_timestamp = serializers.DateTimeField()