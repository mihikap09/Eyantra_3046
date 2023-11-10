import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the pre-trained image classifier model
model_path = "C://Users//HP//vgg16_model_changed.keras"  # Replace with the actual path to your model
classifier_model = load_model(model_path)

# Load the labels for the classes
labels_path = "C://task_2c_eyantra//labels.txt"  # Replace with the actual path to your labels file
with open(labels_path, 'r') as file:
    labels = file.read().splitlines()

# Folder containing the ROIs
roi_folder = "roi_images"

# Get a list of all files in the ROI folder
roi_files = [f for f in os.listdir(roi_folder) if os.path.isfile(os.path.join(roi_folder, f))]

# Iterate through each ROI file and make predictions
for roi_file in roi_files:
    # Load the ROI image
    roi_path = os.path.join(roi_folder, roi_file)
    roi_img = image.load_img(roi_path, target_size=(224, 224))
    roi_array = image.img_to_array(roi_img)
    roi_array = np.expand_dims(roi_array, axis=0)
    roi_array = preprocess_input(roi_array)

    # Make predictions
    predictions = classifier_model.predict(roi_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the corresponding label from the labels list
    predicted_label = labels[predicted_class_index]

    # Print the result
    print(f"\nPredicted class for {roi_file}: {predicted_label}")

    # Display the ROI image
    cv2.imshow(f'ROI: {roi_file}', cv2.imread(roi_path))
    cv2.waitKey(0)

cv2.destroyAllWindows()







