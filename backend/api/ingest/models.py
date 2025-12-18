from django.db import models

# Create your models here.
class IngestData(models.Model):
    pg_no = models.IntegerField()