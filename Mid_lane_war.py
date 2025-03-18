import cv2
import numpy as np
from collections import deque

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

FOCAL_LENGTH = 700
REAL_WIDTH_PERSON = 40  # in cm

cap = cv2.VideoCapture(0)

# Smoothing buffers (deque for moving average)
distance_buffer = deque(maxlen=5)  # Adjust window size for smoother effect
error_buffer = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed error.")
        break

    (h, w) = frame.shape[:2]
    center_x = w // 2

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    for (x, y, bw, bh), weight in zip(boxes, weights):
        if weight > 0.5:
            object_center_x = x + bw // 2
            error = object_center_x - center_x

            # Distance estimation
            distance = (REAL_WIDTH_PERSON * FOCAL_LENGTH) / bw

            # Store values for smoothing
            distance_buffer.append(distance)
            error_buffer.append(error)

            # Smooth values using moving average
            smooth_distance = round(np.mean(distance_buffer), 2)
            smooth_error = round(np.mean(error_buffer), 2)

            # Adjustment suggestion logic
            if abs(smooth_error) < 20:
                adjustment = "Centered"
            elif smooth_error < 0:
                adjustment = "Move Right"
            else:
                adjustment = "Move Left"

            # PRINT to console
            print(f"\n[Detection]")
            print(f"  Smoothed Distance: {smooth_distance} mm")
            print(f"  Smoothed Center Error: {smooth_error} px")
            print(f"  Suggestion: {adjustment}")

            # GUI Overlay
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {smooth_distance} mm", (x, y + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Alignment: {adjustment}", (x, y + bh + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Error: {smooth_error} px", (x, y + bh + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    cv2.line(frame, (center_x, 0), (center_x, h), (255, 0, 0), 2)
    cv2.imshow("Stable Detection & Distance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
