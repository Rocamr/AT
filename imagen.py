import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv

load_dotenv()

# Credenciales de Azure
face_api_key = os.getenv('FACE_API_KEY')
face_api_endpoint = os.getenv('FACE_API_ENDPOINT')

# Cliente para Azure Face API
face_client = FaceClient(face_api_endpoint, CognitiveServicesCredentials(face_api_key))

def recognize_face(image_path):
    with open(image_path, 'rb') as image_stream:
        detected_faces = face_client.face.detect_with_stream(image_stream)
        return detected_faces
