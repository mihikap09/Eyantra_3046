import numpy as np
import cv2
from cv2 import aruco
import math

if __name__ == "__main__":

   # Load the ArUco dictionary (choose an appropriate dictionary)
   img_dir_path = "public_test_cases/"
   marker = 'aruco'

   for file_num in range(0,2):
        img_file_path = img_dir_path +  marker + '_' + str(file_num) + '.png'

        # read image using opencv
        img = cv2.imread(img_file_path)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)


# Create ArUco parameters
    
        parameters = cv2.aruco.DetectorParameters()
    
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load the image
        

# Detect markers in the image
    
        corners, ids, rejected = detector.detectMarkers(img)

        if ids is not None:
            for i in range(len(ids)):
                # Get the four corner coordinates of the marker
                marker_corners = corners[i][0]

                # Calculate the center coordinates of the marker
                cx = int((marker_corners[0][0] + marker_corners[2][0]) / 2)
                cy = int((marker_corners[0][1] + marker_corners[2][1]) / 2)

                # Calculate the angle of the marker
                angle = math.degrees(math.atan2(marker_corners[1][1] - marker_corners[0][1],
                                                marker_corners[1][0] - marker_corners[0][0]))

                print(f"ID: {ids[i][0]}")
                print(f"Marker Corners: {marker_corners}")
                print(f"Center Coordinates (x, y): ({cx}, {cy})")
                print(f"Angle: {angle} degrees")

    # Draw the detected markers on the image
   
        img = cv2.aruco.drawDetectedMarkers(img, corners, ids)

# Display the image
        cv2.imshow('ArUco Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()









   
    
    
