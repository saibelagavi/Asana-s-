import cv2
import easyocr
import numpy as np

def get_mix_strings(item):
    for ch in item:
        if not ch.isalpha():
            if not ch.isdigit():
                return False
    return True

def get_all_num(item):
    return all([ch.isdigit() for ch in item])

def get_all_alpha(item):
    return all([ch.isalpha() for ch in item])

def get_color_mask(image, color):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for the color
    if color == "green":
        lower = np.array([40, 40, 40])
        upper = np.array([70, 255, 255])
    elif color == "skyblue":
        lower = np.array([90, 50, 50])
        upper = np.array([120, 255, 255])
    elif color == "yellow":
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
    elif color == "pink":
        lower = np.array([140, 100, 100])
        upper = np.array([170, 255, 255])
    elif color == "red":
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
    else:
        print("Invalid color specified.")
        return None

    # Create mask for the specified color range
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask

def OCR_header(color_mask, reader):
    result = reader.readtext(color_mask, detail=0, text_threshold=0.90)
    final_header = []
    for item in result:
        if get_all_num(item):
            continue
        elif get_mix_strings(item):
            final_header.append(item)
        elif get_all_alpha(item):
            final_header.append((item))
            
    return final_header

def OCR_values(color_mask, reader):
    result = reader.readtext(color_mask, detail=0, text_threshold=0.90)
    value = [x for x in result if any(x1.isdigit() for x1 in x)]
    return value

def extract_features(image_path):
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])

    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Perform color segmentation and OCR
    features = {}

    # Header Colors
    header_colors = {"green": "heart rate", "skyblue": "Spo2", "yellow": "Respiratory", "pink": "temperature", "red": "NIBP"}
    for color, feature in header_colors.items():
        color_mask = get_color_mask(image, color)
        header_text = OCR_header(color_mask, reader)
        features[feature + "_header"] = header_text

    # Value Colors
    value_colors = {"green": "heart rate", "skyblue": "Spo2", "yellow": "Respiratory", "pink": "temperature", "red": "NIBP"}
    for color, feature in value_colors.items():
        color_mask = get_color_mask(image, color)
        value_text = OCR_values(color_mask, reader)
        features[feature + "_values"] = value_text

    return features

def print_features(features):
    for key, value in features.items():
        print(f"{key}: {value}")

# Path to the input image
image_path = 'D:/asana/1.jpeg'

# Extract features from the input image
features = extract_features(image_path)

# Print extracted features
print_features(features)
