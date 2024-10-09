from django.shortcuts import render
# src/mymodelapi/predictions/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .model import model
from .serializers import PredictionInputSerializer
import numpy as np

class PredictionView(APIView):
    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)
        if serializer.is_valid():
            # Extract features from the validated data
            features = np.array([[serializer.validated_data['feature1'],
                                  serializer.validated_data['feature2']]])
            # Make prediction
            prediction = model.predict(features)
            return Response({"prediction": prediction[0]}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

