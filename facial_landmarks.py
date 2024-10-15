import cv2
import mediapipe as mp

# Initialize Mediapipe face mesh model
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

def extract_facial_landmarks(image_path):
    """Extracts facial landmarks using Mediapipe."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image from path: {image_path}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmarks_list = [(int(l.x * img.shape[1]), int(l.y * img.shape[0])) for l in landmarks.landmark]
            return landmarks_list
        else:
            return None
