import json
from face_comparison import compare_photos_with_explanations

if __name__ == '__main__':
    reference_image_path = './photos/img-1.jpg'
    live_image_path = './photos/srk2.jpg'
    
    result = compare_photos_with_explanations(reference_image_path, live_image_path)
    print(json.dumps(result, indent=4))
