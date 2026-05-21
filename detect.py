from deepface import DeepFace

img_path = "person.jpg"

# Detect face
result = DeepFace.extract_faces(img_path)

print("Faces detected:", len(result))