from rest_framework import serializers

class CommentsSerializer(serializers.Serializer):
    id = serializers.CharField()
    comment = serializers.CharField()
    author = serializers.CharField()
    p_timestamp = serializers.DateTimeField()
    t_timestamp = serializers.DateTimeField()
    language = serializers.CharField(required=False, allow_null=True)
    cleaned_text = serializers.CharField(required=False, allow_null=True)

    def __init__(self, *args, **kwargs):
        mode = kwargs.pop("mode", None)  # "preview" or "sep"
        super().__init__(*args, **kwargs)

        if mode == "preview":
            # keep only preview fields
            keep = {"id", "comment", "author", "p_timestamp", "t_timestamp"}
            for name in list(self.fields):
                if name not in keep:
                    self.fields.pop(name)
        elif mode == "sep":
            # keep all fields (default behavior)
            pass