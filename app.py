import cv2
import numpy as np
import torch
import time
import easyocr

# Load YOLOv5 model for car detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Create an EasyOCR reader object for license plate detection
reader = easyocr.Reader(['en'])

# Define parking space positions (manually set based on your camera view)
parking_spaces = [
    ((50, 100), (200, 250)),  # Parking space 1
    ((250, 100), (400, 250)),  # Parking space 2
    ((450, 100), (600, 250)),  # Parking space 3
]

# To track the occupancy status of each space
parking_status = [None, None, None]  # None means the space is available initially

def detect_cars(frame):
    """
    Detect cars in the frame using YOLOv5.
    """
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Extract bounding boxes

    car_boxes = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_name = model.names[int(cls)]
        if class_name in ['car', 'truck', 'bus'] and conf > 0.5:
            car_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return car_boxes

def extract_license_plate(frame, car_box):
    """
    Extract license plate number from a single car box in the frame.
    """
    x1, y1, x2, y2 = car_box
    # Crop the region of interest (ROI) containing the car's plate
    car_roi = frame[y1:y2, x1:x2]
    # Use EasyOCR to extract text from the plate area
    ocr_results = reader.readtext(car_roi)

    # Filter the results for license plate-like text
    for (bbox, text, _) in ocr_results:
        if len(text) > 3:  # Assuming plate numbers have more than 3 characters
            return text
    return None

def log_parking_status(space_id, status, timestamp, plate_number=None):
    """
    Log parking status (occupied or available) and plate number to a text file.
    """
    with open("parking_log.txt", "a") as log_file:
        if status == "Occupied" and plate_number:
            log_file.write(f"Parking Space {space_id + 1}: {status}, Plate Number: {plate_number} at {timestamp}\n")
        else:
            log_file.write(f"Parking Space {space_id + 1}: {status} at {timestamp}\n")

def check_parking_availability(frame, car_boxes):
    """
    Check if parking spaces are occupied and log status with plate number if occupied.
    """
    for i, (top_left, bottom_right) in enumerate(parking_spaces):
        space_status = "Available"
        color = (0, 255, 0)  # Green for available
        plate_number = None

        for (x1, y1, x2, y2) in car_boxes:
            if x1 < bottom_right[0] and x2 > top_left[0] and y1 < bottom_right[1] and y2 > top_left[1]:
                space_status = "Occupied"
                color = (0, 0, 255)  # Red for occupied
                # Extract the license plate number from the car in the parking space
                plate_number = extract_license_plate(frame, (x1, y1, x2, y2))
                break

        # Log parking status if it changes
        if parking_status[i] != space_status:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_parking_status(i, space_status, timestamp, plate_number)
            parking_status[i] = space_status  # Update the status for this space

        # Draw parking space rectangle and status
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.putText(frame, f"Space {i + 1}: {space_status}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    """
    Main function to run the smart parking management system.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_resized = cv2.resize(frame, (640, 480))

        # Detect cars
        car_boxes = detect_cars(frame_resized)

        # Check parking availability and log status
        frame_with_status = check_parking_availability(frame_resized, car_boxes)

        # Display the frame
        cv2.imshow("Smart Parking Management", frame_with_status)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
