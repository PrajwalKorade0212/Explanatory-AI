import cv2
from skimage.feature import hog
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import shap
# Function to extract HOG features from an image
def extract_hog_features(image):
    # Resize the image to a consistent size
    resized_image = cv2.resize(image, (64, 64))  # Adjust the size as needed
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features and their visualization
    features = hog(gray_image, orientations=8, pixels_per_cell=(2, 2),
                   cells_per_block=(1, 1), visualize=False)
    
    return features
# Function to load and label images
def load_and_label_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(".jpeg"):
            # Load the image
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Error loading image: {img_path}")
                continue
            
            # Extract HOG features
            hog_features = extract_hog_features(image)
            
            # Append features and labels to the lists
            images.append(hog_features)
            labels.append(label)
    
    return images, labels
# Specify the paths to the "pneumonia" and "normal" folders
pneumonia_folder = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/dataset/pneumonia"
normal_folder = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/dataset/normal"
# Load and label images for pneumonia
pneumonia_images, pneumonia_labels = load_and_label_images(pneumonia_folder, 1)

# Load and label images for normal
normal_images, normal_labels = load_and_label_images(normal_folder, 0)
# Combine pneumonia and normal data
all_images = pneumonia_images + normal_images
all_labels = pneumonia_labels + normal_labels

# Convert the lists of features and labels into numpy arrays
X = np.array(all_images)
y = np.array(all_labels)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
# Evaluate the model on the test set
test_predictions = random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
