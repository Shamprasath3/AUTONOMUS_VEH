import cv2
import numpy as np
import time
from simple_pid import PID
from ultralytics import YOLO

# === Parameters ===
STOP_DISTANCE = 30.0  # Minimum distance before stopping (cm)
KNOWN_WIDTH_OBJECT = 1.0  # Width of the smallest object in cm
FOCAL_LENGTH = 500  # Adjust this value after calibration
MAX_DISTANCE = 100.0  # Maximum distance to detect objects in cm
SAFE_DISTANCE = 50.0  # Safe following distance in cm
MAX_SPEED = 80.0  # Maximum speed (km/h)
MIN_SPEED = 10.0  # Minimum speed to keep moving

# === PID Controllers ===
speed_pid = PID(0.5, 0.1, 0.05, setpoint=SAFE_DISTANCE)
steering_pid = PID(0.4, 0.1, 0.05, setpoint=0)
steering_pid.output_limits = (-26, 26)

# Load YOLO model
model = YOLO("yolov8n.pt")

# === Camera Initialization ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected.")
    exit()

# === Background Subtractor ===
back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)


def estimate_distance(bbox_width, focal_length=700, real_width=1.8):
    """Estimate distance based on bounding box width."""
    return (real_width * focal_length) / (bbox_width + 1e-6)


def determine_lane(x_center, frame_width):
    """Determine which lane the object is in."""
    lane_width = frame_width // 3
    if x_center < lane_width:
        return "Left"
    elif x_center < 2 * lane_width:
        return "Center"
    else:
        return "Right"


def get_safety_lane(current_lane):
    """Determine the safest lane to move to."""
    if current_lane == "Left":
        return "Move to Center or Right"
    elif current_lane == "Center":
        return "Move to Left or Right"
    else:
        return "Move to Center or Left"


print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    fg_mask = back_sub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    results = model(frame)
    detected_objects = []
    closest_distance = MAX_DISTANCE + 1
    closest_contour = None

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            bbox_width = x2 - x1

            distance = estimate_distance(bbox_width)
            lane = determine_lane(x_center, frame_width)
            safety_lane = get_safety_lane(lane)

            detected_objects.append((x1, y1, x2, y2, distance, lane, safety_lane))

            color = (0, 255, 0) if distance > 5 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{distance:.1f}m, {lane}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, safety_lane, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            if distance < closest_distance:
                closest_distance = distance

    lane_status = "Left lane clear ✅" if closest_distance > SAFE_DISTANCE else "Maintaining Safe Distance 🟡" if closest_distance > STOP_DISTANCE else "Both lanes blocked ❌"
    steering_angle = steering_pid(-1 if "Left" in lane_status else 1)
    speed_adjustment = speed_pid(closest_distance)
    current_speed = max(MIN_SPEED, min(MAX_SPEED, MAX_SPEED - speed_adjustment))
    brake_pressure = max(0, min(100, (SAFE_DISTANCE - closest_distance) / 5))
    fuel_reduction = min(100, brake_pressure)
    engine_temp = min(110, 90 + (brake_pressure / 2))
    brake_temp = min(100, 30 + (brake_pressure / 1.5))

    print(f"\n📏 Distance to Obstacle: {closest_distance:.1f} cm")
    print(f"🛣 Lane Status: {lane_status}")
    print(f"➡ Steering Angle: {steering_angle:.1f}°")
    print(f"🚗 Speed: {current_speed:.1f} km/h | 🛑 Distance: {closest_distance} cm")
    print(f"🛞 Brake Pressure: {brake_pressure:.1f}% | ⛽ Fuel Flow Reduction: {fuel_reduction:.1f}%")
    print(f"🔥 Brake Temperature: {brake_temp:.1f}°C | 🔧 Engine Temp: {engine_temp:.1f}°C")

    cv2.putText(frame, f"Distance: {closest_distance:.1f} cm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Lane: {lane_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {current_speed:.1f} km/h", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Steering: {steering_angle:.1f}°", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("AI Vehicle Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()