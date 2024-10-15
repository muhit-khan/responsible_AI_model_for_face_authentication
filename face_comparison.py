import os
import shutil
import tempfile
import numpy as np
from deepface import DeepFace
from facial_landmarks import extract_facial_landmarks
from utils.helpers import calculate_curve_length

def compare_photos_with_explanations(reference_image_path, live_image_path, threshold=60):
    """
    Compares two photos, extracting facial landmarks and calculating similarities 
    between key features (e.g., eyes, nose, lips). It also uses DeepFace for identity verification.
    
    Args:
        reference_image_path (str): Path to the reference image.
        live_image_path (str): Path to the live image for comparison.
        threshold (int): Minimum percentage similarity to declare a responsible identity match.
    
    Returns:
        dict: A dictionary containing the identity verification, feature similarities, and explanations.
        or
        tuple: (False, error message) if an exception occurs or landmarks cannot be extracted.
    """
    try:
        # Create a temporary directory for uploads
        with tempfile.TemporaryDirectory() as upload_dir:
            # Copy reference and live images into the temporary directory
            reference_temp_path = os.path.join(upload_dir, 'reference_image.jpg')
            live_temp_path = os.path.join(upload_dir, 'live_image.jpg')

            # Copy the input images to the temporary paths
            shutil.copy(reference_image_path, reference_temp_path)
            shutil.copy(live_image_path, live_temp_path)

            # Verify whether the two images belong to the same person using DeepFace
            result = DeepFace.verify(img1_path=reference_temp_path, img2_path=live_temp_path)
            identity = result['verified']  # Result from DeepFace

            # Extract facial landmarks
            reference_landmarks = extract_facial_landmarks(reference_temp_path)
            live_landmarks = extract_facial_landmarks(live_temp_path)

            # Check if landmarks could be extracted from both images
            if live_landmarks is None:
                return False, 'Could not extract landmarks from live image'
            if reference_landmarks is None:
                return False, 'Could not extract landmarks from reference image'

            # Initialize lists for storing feature explanations and similarity scores
            feature_explanations = []
            similarity_scores = []

            # Calculate similarity for different facial features (e.g., eye distance, nose width, etc.)
            # Example: Distance between eyes
            reference_eye_distance = np.linalg.norm(np.array(reference_landmarks[33]) - np.array(reference_landmarks[263]))
            live_eye_distance = np.linalg.norm(np.array(live_landmarks[33]) - np.array(live_landmarks[263]))
            eye_distance_similarity = 100 - abs(reference_eye_distance - live_eye_distance) / reference_eye_distance * 100
            feature_explanations.append(f"Eye distance similarity: {eye_distance_similarity:.2f}%")
            similarity_scores.append(eye_distance_similarity)

            # Nose width
            reference_nose_width = np.linalg.norm(np.array(reference_landmarks[197]) - np.array(reference_landmarks[2]))
            live_nose_width = np.linalg.norm(np.array(live_landmarks[197]) - np.array(live_landmarks[2]))
            nose_width_similarity = 100 - abs(reference_nose_width - live_nose_width) / reference_nose_width * 100
            feature_explanations.append(f"Nose width similarity: {nose_width_similarity:.2f}%")
            similarity_scores.append(nose_width_similarity)

            # Nose height
            reference_nose_height = np.linalg.norm(np.array(reference_landmarks[1]) - np.array(reference_landmarks[2]))
            live_nose_height = np.linalg.norm(np.array(live_landmarks[1]) - np.array(live_landmarks[2]))
            nose_height_similarity = 100 - abs(reference_nose_height - live_nose_height) / reference_nose_height * 100
            feature_explanations.append(f"Nose height similarity: {nose_height_similarity:.2f}%")
            similarity_scores.append(nose_height_similarity)

            # Lip width
            reference_lip_width = np.linalg.norm(np.array(reference_landmarks[61]) - np.array(reference_landmarks[291]))
            live_lip_width = np.linalg.norm(np.array(live_landmarks[61]) - np.array(live_landmarks[291]))
            lip_width_similarity = 100 - abs(reference_lip_width - live_lip_width) / reference_lip_width * 100
            feature_explanations.append(f"Lip width similarity: {lip_width_similarity:.2f}%")
            similarity_scores.append(lip_width_similarity)

            # Jawline width
            reference_jawline_width = np.linalg.norm(np.array(reference_landmarks[152]) - np.array(reference_landmarks[377]))
            live_jawline_width = np.linalg.norm(np.array(live_landmarks[152]) - np.array(live_landmarks[377]))
            jawline_width_similarity = 100 - abs(reference_jawline_width - live_jawline_width) / reference_jawline_width * 100
            feature_explanations.append(f"Jawline width similarity: {jawline_width_similarity:.2f}%")
            similarity_scores.append(jawline_width_similarity)

            # Eyebrow distance
            reference_eyebrow_distance = np.linalg.norm(np.array(reference_landmarks[70]) - np.array(reference_landmarks[300]))
            live_eyebrow_distance = np.linalg.norm(np.array(live_landmarks[70]) - np.array(live_landmarks[300]))
            eyebrow_distance_similarity = 100 - abs(reference_eyebrow_distance - live_eyebrow_distance) / reference_eyebrow_distance * 100
            feature_explanations.append(f"Eyebrow distance similarity: {eyebrow_distance_similarity:.2f}%")
            similarity_scores.append(eyebrow_distance_similarity)

            # Eyebrow curve length
            reference_eyebrow_curve_length = calculate_curve_length(reference_landmarks, [70, 63, 105, 66, 107, 55, 193])
            live_eyebrow_curve_length = calculate_curve_length(live_landmarks, [70, 63, 105, 66, 107, 55, 193])
            eyebrow_curve_similarity = 100 - abs(reference_eyebrow_curve_length - live_eyebrow_curve_length) / reference_eyebrow_curve_length * 100
            feature_explanations.append(f"Eyebrow curve length similarity: {eyebrow_curve_similarity:.2f}%")
            similarity_scores.append(eyebrow_curve_similarity)

            # Forehead height
            reference_forehead_height = np.linalg.norm(np.array(reference_landmarks[10]) - np.array(reference_landmarks[151]))
            live_forehead_height = np.linalg.norm(np.array(live_landmarks[10]) - np.array(live_landmarks[151]))
            forehead_height_similarity = 100 - abs(reference_forehead_height - live_forehead_height) / reference_forehead_height * 100
            feature_explanations.append(f"Forehead height similarity: {forehead_height_similarity:.2f}%")
            similarity_scores.append(forehead_height_similarity)

            # Forehead width
            reference_forehead_width = np.linalg.norm(np.array(reference_landmarks[10]) - np.array(reference_landmarks[338]))
            live_forehead_width = np.linalg.norm(np.array(live_landmarks[10]) - np.array(live_landmarks[338]))
            forehead_width_similarity = 100 - abs(reference_forehead_width - live_forehead_width) / reference_forehead_width * 100
            feature_explanations.append(f"Forehead width similarity: {forehead_width_similarity:.2f}%")
            similarity_scores.append(forehead_width_similarity)

            # Eye to mouth distance
            reference_eye_mouth_distance = np.linalg.norm(np.array(reference_landmarks[33]) - np.array(reference_landmarks[13]))
            live_eye_mouth_distance = np.linalg.norm(np.array(live_landmarks[33]) - np.array(live_landmarks[13]))
            eye_mouth_similarity = 100 - abs(reference_eye_mouth_distance - live_eye_mouth_distance) / reference_eye_mouth_distance * 100
            feature_explanations.append(f"Eye to mouth distance similarity: {eye_mouth_similarity:.2f}%")
            similarity_scores.append(eye_mouth_similarity)

            # Nose to chin distance
            reference_nose_chin_distance = np.linalg.norm(np.array(reference_landmarks[2]) - np.array(reference_landmarks[152]))
            live_nose_chin_distance = np.linalg.norm(np.array(live_landmarks[2]) - np.array(live_landmarks[152]))
            nose_chin_similarity = 100 - abs(reference_nose_chin_distance - live_nose_chin_distance) / reference_nose_chin_distance * 100
            feature_explanations.append(f"Nose to chin distance similarity: {nose_chin_similarity:.2f}%")
            similarity_scores.append(nose_chin_similarity)

            # Calculate the average similarity score
            average_similarity = sum(similarity_scores) / len(similarity_scores)

            # Determine if the average similarity meets the threshold
            responsible_identity = bool(average_similarity >= threshold)

            # Build explanation result
            explanation_result = {
                'identity': identity,  # Result from DeepFace
                'responsible_identity': responsible_identity,  # Controlled by the threshold
                'average_similarity': average_similarity,
                'explanations': feature_explanations
            }

            # The temporary directory and its contents will automatically be cleaned up here
            return explanation_result

    except Exception as e:
        print(f"Error during face comparison: {e}")
        return False, f"Error during face comparison: {e}"
