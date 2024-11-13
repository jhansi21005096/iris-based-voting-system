# iris-based-voting-system
```
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
```

# Load VGG16 model for feature extraction
```
def load_vgg16_model():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return model
```

# Preprocess the image
```
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
```

# Extract features using the pre-trained VGG16 model
```
def extract_features(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    features = model.predict(preprocessed_image)
    flattened_features = features.flatten()  # Flatten the features for comparison
    return flattened_features
```
# Load dataset and create embeddings for each registered voter
```
def create_database(model, dataset_path):
    database = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            images = os.listdir(person_folder)
            # Use the first image as the stored feature vector
            if images:
                image_path = os.path.join(person_folder, images[0])
                features = extract_features(model, image_path)
                database[person_name] = features
    return database
```
# Verify new image against stored database
```
def verify_iris(model, database, new_image_path, threshold=0.9):
    new_image_features = extract_features(model, new_image_path)
    for person_name, stored_features in database.items():
        similarity = cosine_similarity([new_image_features], [stored_features])[0][0]
        if similarity >= threshold:
            print(f"Authentication successful for {person_name}.")
            return True
    print("Authentication failed.")
    return False
```
# Paths
```
dataset_path = './iris_vgg'  # Path to the dataset folder
new_image_path = './new_iris_image.jpg'  # Path to a new iris image to authenticate
```
# Main Script
```
model = load_vgg16_model()
database = create_database(model, dataset_path)
verify_iris(model, database, new_image_path)
```

# Output
![out4e](https://github.com/user-attachments/assets/2ba0ccf5-3c77-4d20-8dfa-64ff0160986a)
![out5e](https://github.com/user-attachments/assets/ce729bbc-9218-47de-a3a6-7e5ac7fe72c4)
![out6e](https://github.com/user-attachments/assets/76f3db78-ec2a-4a1c-b489-c5161855af19)
![out7e](https://github.com/user-attachments/assets/f3d0b0d6-2f31-40fe-bb06-9053476eecca)



