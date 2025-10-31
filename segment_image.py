# Hand Segmentation Script

import os
import cv2
import numpy as np
from tqdm import tqdm

INPUT_ROOT = "dataset_splitted"
OUTPUT_ROOT = "dataset_segmented"

# Image size
IMG_SIZE = (224, 224)


def create_output_structure():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:
        split_path = os.path.join(INPUT_ROOT, split)
        if os.path.exists(split_path):
            for class_name in os.listdir(split_path):
                class_in = os.path.join(split_path, class_name)
                if os.path.isdir(class_in):
                    class_out = os.path.join(OUTPUT_ROOT, split, class_name)
                    os.makedirs(class_out, exist_ok=True)


# -----------------------------
# Segmentation function
# -----------------------------
def segment_hand(image):
    """
    Improved HSV-based hand segmentation.
    - Wider HSV range to handle lighting & skin tone variations
    - Morphological operations for cleaner mask
    - Contour filtering to keep only the largest region (hand)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV range for skin tones
    lower_skin = np.array([0, 40, 70], dtype=np.uint8)
    upper_skin = np.array([25, 155, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mask = clean_mask

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply mask to original image
    segmented = cv2.bitwise_and(image, image, mask=mask)

    return segmented


# -----------------------------
# Process dataset
# -----------------------------
def process_dataset():
    create_output_structure()

    for split in ["train", "val", "test"]:
        split_input = os.path.join(INPUT_ROOT, split)
        split_output = os.path.join(OUTPUT_ROOT, split)

        if not os.path.exists(split_input):
            continue

        for class_name in tqdm(os.listdir(split_input), desc=f"Processing {split}"):
            class_input = os.path.join(split_input, class_name)
            class_output = os.path.join(split_output, class_name)

            if not os.path.isdir(class_input):
                continue

            for filename in os.listdir(class_input):
                input_path = os.path.join(class_input, filename)
                output_path = os.path.join(class_output, filename)

                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Skipping unreadable image: {input_path}")
                    continue

                # Resize to ensure consistent size
                img = cv2.resize(img, IMG_SIZE)

                # Segment hand
                segmented_img = segment_hand(img)

                # Save output (same filename)
                cv2.imwrite(output_path, segmented_img)


# -----------------------------
# Run the script
# -----------------------------
if __name__ == "__main__":
    print("Starting segmentation process...")
    process_dataset()
    print(f"Segmentation completed! Check: {OUTPUT_ROOT}")
