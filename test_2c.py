import cv2
import numpy as np
import os 



# Create a folder to organize the ROIs
output_folder = "roi_images"
os.makedirs(output_folder, exist_ok=True)  # Creates the folder if it doesn't exist
# Load the PNG image
image = cv2.imread("C://task_2c_eyantra//sample.png")

# Convert the image to grayscale (required for ArUco marker detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect ArUco markers
corners, ids, rejected = detector.detectMarkers(image)

# Draw rectangles around detected markers and obtain their coordinates
if ids is not None:
    for i in range(len(ids)):
        # Draw a rectangle around the detected marker
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Obtain the coordinates of the marker
    for i in range(len(ids)):
        marker_id = ids[i][0]
        marker_corners = corners[i][0]
        x, y = marker_corners.mean(axis=0)  # Calculate the center of the marker

        # Print the marker ID and coordinates
        print(f"Marker ID {marker_id}: X={x}, Y={y}")

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
    cv2.rectangle(image, (x, y), (x + width, y + height), rect_color, rect_thickness)

    # Extract the region of interest (ROI)
    roi = image[y:y + height, x:x + width]

    # Save the ROI as a separate image file
    roi_filename =  os.path.join(output_folder, f"roi_{counter}.png")
    cv2.imwrite(roi_filename, roi)

    # TODO: Pass 'roi_filename' to your classifier for further processing

    # Increment the counter for the next ROI
    counter += 1

# Display the image with drawn rectangles around specified regions
cv2.imshow('Specified Regions of Interest', image)
cv2.waitKey(0)
cv2.destroyAllWindows()










