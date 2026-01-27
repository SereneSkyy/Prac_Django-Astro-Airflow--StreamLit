from rest_framework import serializers

class CommentsSerializer(serializers.Serializer):
    id = serializers.CharField()
    comment = serializers.CharField()
    author = serializers.CharField()
    p_timestamp = serializers.CharField() 
    t_timestamp = serializers.CharField()
    language = serializers.CharField(required=False, allow_null=True)
    cleaned_text = serializers.CharField(required=False, allow_null=True)
    sentiment = serializers.CharField(required=False, allow_null=True) 

    def __init__(self, *args, **kwargs):
        mode = kwargs.pop("mode", None)
        super().__init__(*args, **kwargs)

        if mode == "preview":
            # ADDED "cleaned_text" to this set so it doesn't get deleted
            keep = {"id", "comment", "author", "p_timestamp", "t_timestamp", "language", "sentiment", "cleaned_text"}
            for name in list(self.fields):
                if name not in keep:
                    self.fields.pop(name)