import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os

# Set the path to your pre-trained model
model_path = 'my_trained_model.keras'

# Load the pre-trained model
model = load_model(model_path)

# Define the path to your test data directory
test_data_dir = "C://Users//HP//Downloads//testing"

# List to store test image file paths and corresponding labels
test_images = []
test_labels = []

# Mapping of file names to class labels
filename_to_class = {
    "combat1.jpeg": "combat",
    "combat2.jpeg": "combat",
    "fire1.jpeg": "fire",
    "fire2.jpeg": "fire",
    "militaryvehicles1.jpeg": "military_vehicles",
    "militaryvehicles2.jpeg": "military_vehicles",
    "rehab1.jpeg": "rehab",
    "rehab2.jpeg": "rehab",
    "destroyedbuildings1.jpeg": "destroyed_buildings",
    "destroyedbuildings2.jpeg": "destroyed_buildings",
}

# Walk through the test data directory and collect file paths
for root, dirs, files in os.walk(test_data_dir):
    for file in files:
        if file in filename_to_class:
            # Append the file path
            image_path = os.path.join(root, file)
            test_images.append(image_path)
            # Append the corresponding label based on the mapping
            test_labels.append(filename_to_class[file])

# Perform inference on the test images
predictions = []

for image_path in test_images:
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = tf.expand_dims(img, 0)  # Add batch dimension
    prediction = model.predict(img)
    predictions.append(prediction)

print('done')
# Process the predictions and true labels as needed

# Calculate accuracy (You need to define this logic based on your labels)
# accuracy = ...

# Print or use the accuracy
from sklearn.metrics import accuracy_score

# Process the predictions and true labels as needed

# Convert predictions to class labels (class with the highest probability)
predicted_labels = []

# Convert predictions to class labels (class with the highest probability)
for prediction in predictions:
    predicted_class_index = np.argmax(prediction)
    # Convert the index to a class label
    predicted_class_label = None
    for key, value in filename_to_class.items():
        if value == predicted_class_index:
            predicted_class_label = key
            break
    if predicted_class_label:
        predicted_labels.append(predicted_class_label)
    else:
        # Handle the case where the class index doesn't match any known labels
        predicted_labels.append("Unknown")
# Get the true labels from the list
true_labels = test_labels

# Calculate accuracy using sklearn's accuracy_score function
accuracy = accuracy_score(true_labels, predicted_labels)

# Print or use the accuracy
print(f"Test Accuracy: {accuracy}")









