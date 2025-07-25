import os
import cv2
import numpy as np
from tqdm import tqdm

# Set dataset path
DATASET_PATH = 'dataset'

images = []
labels = []

# Loop through each person's folder with progress bar
for person in tqdm(os.listdir(DATASET_PATH), desc="Processing persons"):
    person_folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    for img_name in tqdm(os.listdir(person_folder), desc=f"Processing {person}", leave=False):
        img_path = os.path.join(person_folder, img_name)

        # Read and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize (0 to 1)

        images.append(img)
        labels.append(person)

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

np.save("preprocessed/X.npy", X)
np.save("preprocessed/y.npy", y)
