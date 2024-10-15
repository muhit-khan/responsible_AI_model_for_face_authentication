import numpy as np

def calculate_curve_length(landmarks, indices):
    curve_length = 0
    for i in range(len(indices) - 1):
        curve_length += np.linalg.norm(np.array(landmarks[indices[i]]) - np.array(landmarks[indices[i + 1]]))
    return curve_length
