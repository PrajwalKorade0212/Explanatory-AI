#!/usr/bin/env python
# coding: utf-8

# In[17]:


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


# In[18]:


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

    


# In[19]:


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


# In[20]:


# Specify the paths to the "pneumonia" and "normal" folders
pneumonia_folder = "C:/Users/lokis/OneDrive/Desktop/Explanatory-AI-main/Explanatory-AI-main/dataset/pneumonia"
normal_folder = "C:/Users/lokis/OneDrive/Desktop/Explanatory-AI-main/Explanatory-AI-main/dataset/normal"


# In[22]:



# Load and label images for pneumonia
pneumonia_images, pneumonia_labels = load_and_label_images(pneumonia_folder, 1)


# In[23]:


# Load and label images for normal
normal_images, normal_labels = load_and_label_images(normal_folder, 0)
# Combine pneumonia and normal data
all_images = pneumonia_images + normal_images
all_labels = pneumonia_labels + normal_labels


# In[24]:


# Convert the lists of features and labels into numpy arrays
X = np.array(all_images)
y = np.array(all_labels)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


# In[26]:


# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[27]:


# Create a dummy image with the values [30, 87000]
dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
dummy_image[:, :, 0] = 30
dummy_image[:, :, 1] = 87000

# Extract HOG features from the dummy image
new_data_point_hog = extract_hog_features(dummy_image)

# Scale the new data point using the same StandardScaler
scaled_new_data_point = sc.transform([new_data_point_hog])

# Predict the result
print(classifier.predict(scaled_new_data_point))


# In[28]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[29]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




