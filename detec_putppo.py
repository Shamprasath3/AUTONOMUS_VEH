import cv2
import numpy as np
import time
from simple_pid import PID
from ultralytics import YOLO
import math

# === Parameters ===
STOP_DISTANCE = 30.0  # cm
KNOWN_WIDTH_OBJECT = 1.0  # cm
FOCAL_LENGTH = 500
MAX_DISTANCE = 100.0  # cm
SAFE_DISTANCE = 50.0  # cm
MAX_SPEED = 80.0  # km/h
MIN_SPEED = 10.0  # km/h
LOOKAHEAD_DISTANCE = 50.0  # cm for pure pursuit

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

# === Car Simulation Parameters ===
car_x, car_y = 320, 480  # Starting position (center bottom of frame)
car_angle = 0  # degrees
wheelbase = 20  # cm


class CarSimulation:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def update(self, speed, steering_angle, dt=0.1):
        # Bicycle model
        beta = math.atan2(steering_angle * math.tan(math.radians(90)), wheelbase)
        self.x += speed * math.cos(math.radians(self.angle)) * dt
        self.y -= speed * math.sin(math.radians(self.angle)) * dt
        self.angle += (speed / wheelbase) * math.tan(math.radians(steering_angle)) * dt


def estimate_distance(bbox_width, focal_length=700, real_width=1.8):
    return (real_width * focal_length) / (bbox_width + 1e-6)


def determine_lane(x_center, frame_width):
    lane_width = frame_width // 3
    if x_center < lane_width:
        return "Left"
    elif x_center < 2 * lane_width:
        return "Center"
    else:
        return "Right"


def get_safety_lane(current_lane):
    if current_lane == "Left":
        return "Move to Center or Right"
    elif current_lane == "Center":
        return "Move to Left or Right"
    else:
        return "Move to Center or Left"


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
    return lines


def pure_pursuit(car, target_x, target_y):
    dx = target_x - car.x
    dy = target_y - car.y
    distance = math.sqrt(dx * dx + dy * dy)

    if distance < 5:  # Target reached
        return 0

    # Calculate desired steering angle
    lookahead_angle = math.degrees(math.atan2(dy, dx))
    angle_error = lookahead_angle - car.angle
    while angle_error > 180: angle_error -= 360
    while angle_error < -180: angle_error += 360

    steering = math.degrees(math.atan2(2 * wheelbase * math.sin(math.radians(angle_error)),
                                       LOOKAHEAD_DISTANCE))
    return max(min(steering, 26), -26)


car = CarSimulation(car_x, car_y, car_angle)

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Lane detection
    lines = detect_lanes(frame)
    mid_lane_warning = False
    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) > 0.5:  # Filter near-horizontal lines
                if slope < 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])

        # Draw lane lines and check mid-lane position
        if left_lines and right_lines:
            left_avg = np.mean(left_lines, axis=0).astype(int)
            right_avg = np.mean(right_lines, axis=0).astype(int)
            cv2.line(frame, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 255), 2)
            cv2.line(frame, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 255), 2)

            lane_center = ((left_avg[0] + right_avg[0]) // 2 + (left_avg[2] + right_avg[2]) // 2) // 2
            mid_lane_warning = abs(car.x - lane_center) > frame_width // 6

    # Object detection
    fg_mask = back_sub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    results = model(frame)
    closest_distance = MAX_DISTANCE + 1
    target_x, target_y = car.x, car.y - LOOKAHEAD_DISTANCE

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            bbox_width = x2 - x1

            distance = estimate_distance(bbox_width)
            lane = determine_lane(x_center, frame_width)
            safety_lane = get_safety_lane(lane)

            color = (0, 255, 0) if distance > 5 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{distance:.1f}m, {lane}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if distance < closest_distance:
                closest_distance = distance
                target_x = x_center
                target_y = (y1 + y2) // 2

    # Control logic
    lane_status = "Left lane clear ✅" if closest_distance > SAFE_DISTANCE else "Maintaining Safe Distance 🟡" if closest_distance > STOP_DISTANCE else "Both lanes blocked ❌"
    steering_angle = pure_pursuit(car, target_x, target_y)
    speed_adjustment = speed_pid(closest_distance)
    current_speed = max(MIN_SPEED, min(MAX_SPEED, MAX_SPEED - speed_adjustment))

    # Update car position
    car.update(current_speed / 36.0, steering_angle)  # Convert km/h to cm/s

    # Draw car (simple rectangle)
    car_length, car_width = 40, 20
    car_corners = [
        (-car_length / 2, -car_width / 2), (car_length / 2, -car_width / 2),
        (car_length / 2, car_width / 2), (-car_length / 2, car_width / 2)
    ]
    rotated_corners = []
    for x, y in car_corners:
        xr = x * math.cos(math.radians(car.angle)) - y * math.sin(math.radians(car.angle)) + car.x
        yr = x * math.sin(math.radians(car.angle)) + y * math.cos(math.radians(car.angle)) + car.y
        rotated_corners.append((int(xr), int(yr)))
    cv2.polylines(frame, [np.array(rotated_corners)], True, (255, 0, 0), 2)

    # Display info
    status_text = "MID-LANE WARNING!" if mid_lane_warning else "On Track"
    cv2.putText(frame, f"Distance: {closest_distance:.1f} cm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Lane: {lane_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {current_speed:.1f} km/h", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Steering: {steering_angle:.1f}°", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, status_text, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if mid_lane_warning else (0, 255, 0), 2)

    cv2.imshow("AI Vehicle Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()