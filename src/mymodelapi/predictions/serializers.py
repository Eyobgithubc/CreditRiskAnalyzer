# src/mymodelapi/predictions/serializers.py
from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    feature1 = serializers.FloatField()
    feature2 = serializers.FloatField()
    
