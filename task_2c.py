'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2C of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2c.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
import imghdr
from sys import platform
import numpy as np
import subprocess
import cv2 as cv    # OpenCV Library
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
arena_path = "C://task_2c_eyantra//arena.png"          # Path of generated arena image
event_list = []
detected_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

# Extracting Events from Arena
def arena_image(arena_path):            # NOTE: This function has already been done for you, don't make any changes in it.
    ''' 
	Purpose:
	---
	This function will take the path of the generated image as input and 
    read the image specified by the path.
	
	Input Arguments:
	---
	`arena_path`: Generated image path i.e. arena_path (declared above) 	
	
	Returns:
	---
	`arena` : [ Numpy Array ]

	Example call:
	---
	arena = arena_image(arena_path)
	'''
    '''
    ADD YOUR CODE HERE
    '''
    frame = cv.imread(arena_path)
    arena = cv.resize(frame, (700, 700))
    print("Success0")
    return arena 
    

def event_identification(arena):        # NOTE: You can tweak this function in case you need to give more inputs 
    ''' 
	Purpose:
	---
	This function will select the events on arena image and extract them as
    separate images.
	
	Input Arguments:
	---
	`arena`: Image of arena detected by arena_image() 	
	
	Returns:
	---
	`event_list`,  : [ List ]
                            event_list will store the extracted event images

	Example call:
	---
	event_list = event_identification(arena)
	'''
    '''
    ADD YOUR CODE HERE
    '''
    # Create a folder to organize the ROIs
    output_folder = "roi_images"
    os.makedirs(output_folder, exist_ok=True)  # Creates the folder if it doesn't exist
    # Load the PNG image
    

    # Convert the image to grayscale (required for ArUco marker detection)
    gray = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)

    # Load the ArUco dictionary and parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(arena)

    # Draw rectangles around detected markers and obtain their coordinates
    if ids is not None:
        for i in range(len(ids)):
            # Draw a rectangle around the detected marker
            cv.aruco.drawDetectedMarkers(arena, corners, ids)

            # Obtain the coordinates of the marker
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i][0]
            x, y = marker_corners.mean(axis=0)  # Calculate the center of the marker

            # Print the marker ID and coordinates
           # print(f"Marker ID {marker_id}: X={x}, Y={y}")

        # Define the coordinates for the regions of interest (x, y, width, height)
    roi_coordinates = [
        (147, 118, 85, 75),
        (132,330,85,75),
        (460,330, 75, 75),
        (450,469,75,67),
        (140,595,75,65 )
        # Add coordinates for other regions as needed
    ]

    # Counter for naming output files
    counter = 1

    # Extract and save the regions of interest
    for coordinates in roi_coordinates:
        x, y, width, height = coordinates

        # Draw a rectangular box around the specified region
        rect_color = (0, 255, 0)  # Green color
        rect_thickness = 2
        cv.rectangle(arena, (x, y), (x + width, y + height), rect_color, rect_thickness)

        # Extract the region of interest (ROI)
        roi = arena[y:y + height, x:x + width]

        # Save the ROI as a separate image file
        roi_filename =  os.path.join(output_folder, f"event_{counter}.png")
        cv.imwrite(roi_filename, roi)
        counter += 1
        # Append the file path to the event_paths list
        event_list.append(roi_filename)

    print("success1")
    return event_list

# Event Detection
def classify_event(image):
    ''' 
	Purpose:
	---
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	Input Arguments:
	---
	`image`: Image path sent by input file 	
	
	Returns:
	---
	`event` : [ String ]
						  Detected event is returned in the form of a string

	Example call:
	---
	event = classify_event(image_path)
	'''
    '''
    ADD YOUR CODE HERE
    '''
    # Load the pre-trained image classifier model

    model_path = "C://test_2c//custom_pretrained_classifier.keras"  # Replace with the actual path to your model
    classifier_model = load_model(model_path)

    # Load the class labels
    class_labels = ["combat", "destroyed_building", "fire", "military_vehicles", "rehab"]

    # Folder containing the ROIs
    #roi_folder = "roi_images"

    # Get a list of all files in the ROI folder
    #roi_files = [f for f in os.listdir(roi_folder) if os.path.isfile(os.path.join(roi_folder, f))]



    # Iterate through each ROI file and make predictions

        # Load the ROI image
        #roi_path = os.path.join(roi_folder, roi_file)
    roi_image = img.load_img(image, target_size=(224, 224))
    roi_array = img.img_to_array(roi_image)
    roi_array = np.expand_dims(roi_array, axis=0)
    roi_array = roi_array/ 255.0  # Normalize pixel values to [0, 1]'''

        # Make predictions
    #predictions = classifier_model.predict(roi_array)

        # Debug prints
    #print(f"\nRaw predictions for {roi_file}: {predictions}")
    
    # Make predictions
    predictions = classifier_model.predict(roi_array)


    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)

        # Debug print
   # print(f"Predicted class index for {roi_file}: {predicted_class_index}")

        # Get the corresponding label from the labels list
    predicted_label = class_labels[predicted_class_index]

        # Print the result
    #print(f"Predicted class for {image}: {predicted_label}:")
        #print the true label

        
    event = predicted_label
    print("success2")
    return event

# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''
###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(event_list):
    for img_index in range(0,5):
        img = event_list[img_index]
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    os.remove('arena.png')
    print("Success3")
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)

def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    input_function()
    #################

    ##### Process #####
    arena = arena_image(arena_path)
    event_list = event_identification(arena)
    detected_list = classification(event_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('arena.png'):
            os.remove('arena.png')
        if os.path.exists('detected_events.txt'):
            os.remove('detected_events.txt')
        sys.exit()
