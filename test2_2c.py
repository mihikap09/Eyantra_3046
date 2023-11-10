import cv2
import numpy as np

# Load the PNG image
image = cv2.imread("C://task_2c_eyantra//sample.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Create ArUco parameters
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Tune detection parameters if necessary
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Loop through the detected contours and filter ArUco markers
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        # Calculate the center coordinates
        x, y, w, h = cv2.boundingRect(approx)
        center_x = x + (w // 2)
        center_y = y + (h // 2)

        # Ensure markers are not too close to the image border
        if (x > 10 and y > 10 and x + w < image.shape[1] - 10 and y + h < image.shape[0] - 10):
            # Detect ArUco markers within the bounding rectangle
            roi = gray[y:y+h, x:x+w]
            corners, ids, _ = detector.detectMarkers(image)

            if ids is not None:
                # Draw a rectangle around the detected marker
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Print the marker IDs and coordinates
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    print(f"Marker ID {marker_id}: X={center_x}, Y={center_y}")

# Display the image with detected markers
cv2.imshow('Detected ArUco Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
