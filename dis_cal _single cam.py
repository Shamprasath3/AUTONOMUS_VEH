import cv2
import numpy as np

# === Parameters ===
KNOWN_WIDTH_OBJECT = 1.0  # Width of the smallest object (e.g., pin) in m
FOCAL_LENGTH = 500        # Adjust this value after calibration
MAX_DISTANCE = 30.0       # Maximum distance to detect objects in m

# === Distance Calculation Function ===
def calculate_distance(known_width, focal_length, per_width):
    """
    Calculate the distance from the object to the camera.
    """
    return (known_width * focal_length) / per_width

# === Camera Initialization ===
cap = cv2.VideoCapture(0)  # Use the default camera

if not cap.isOpened():
    print("Camera not detected.")
    exit()

# === Background Subtractor ===
back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame.")
        break
    
    # === Background Subtraction ===
    fg_mask = back_sub.apply(frame)
    
    # === Noise Reduction ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # === Object Detection and Distance Calculation ===
    closest_distance = MAX_DISTANCE + 1
    closest_contour = None
    
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate distance
            distance = calculate_distance(KNOWN_WIDTH_OBJECT, FOCAL_LENGTH, w)
            
            # Find the closest object within the distance range
            if distance < closest_distance and distance < MAX_DISTANCE:
                closest_distance = distance
                closest_contour = contour
    
    # === Display the closest object and its distance ===
    if closest_contour is not None:
        x, y, w, h = cv2.boundingRect(closest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {closest_distance:.2f} cm", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Object detected at: {closest_distance:.2f} cm")
    
    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close windows
cap.release()
cv2.destroyAllWindows()
