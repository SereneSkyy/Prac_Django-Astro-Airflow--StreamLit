from rest_framework import serializers

class CommentsSerializer(serializers.Serializer):
    id = serializers.CharField()
    comment = serializers.CharField()
    author = serializers.CharField()
    p_timestamp = serializers.DateTimeField()
    t_timestamp = serializers.DateTimeField()
    language = serializers.CharField(required=False, allow_null=True)
    cleaned_text = serializers.CharField(required=False, allow_null=True) # From Notebook    