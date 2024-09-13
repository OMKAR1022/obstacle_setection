import cv2
import depthai as dai
from ultralytics import YOLO

# Function to draw the boxes (left, middle, right)
def draw_boxes(frame):
    height, width, _ = frame.shape

    # Divide the screen into three equal parts
    box_width = width // 3

    # Draw left box (move right if obstacle)
    cv2.rectangle(frame, (0, 0), (box_width, height), (255, 0, 0), 2)  # Blue box

    # Draw middle box (safe zone for walking)
    cv2.rectangle(frame, (box_width, 0), (2 * box_width, height), (0, 255, 0), 2)  # Green box

    # Draw right box (move left if obstacle)
    cv2.rectangle(frame, (2 * box_width, 0), (width, height), (0, 0, 255), 2)  # Red box

    # Display labels
    cv2.putText(frame, 'Left Box (Move Right)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, 'Middle Box (Walking Area)', (box_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, 'Right Box (Move Left)', (2 * box_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Function to perform object detection using YOLOv8
def detect_objects_in_roi(frame, model, roi_box):
    # Crop the ROI from the frame
    x_start, y_start, x_end, y_end = roi_box
    roi = frame[y_start:y_end, x_start:x_end]

    # Perform detection on the ROI
    results = model(roi)

    # Process detection results
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                # Extract class and confidence if available
                class_id = int(boxes.cls.cpu().numpy()[i]) if boxes.cls is not None else -1
                prob = boxes.conf.cpu().numpy()[i] if boxes.conf is not None else 0
                xmin, ymin, xmax, ymax = box
                detections.append([xmin, ymin, xmax, ymax, class_id, prob])

    return detections


# Function to draw bounding boxes around detected objects in the specified zone
def draw_detected_objects(frame, detections, roi_box):
    x_start, y_start, x_end, y_end = roi_box
    for detection in detections:
        xmin, ymin, xmax, ymax, cls, prob = detection
        # Scale bounding boxes from ROI to original frame
        xmin += x_start
        xmax += x_start
        ymin += y_start
        ymax += y_start

        # Draw bounding box and label
        if prob > 0.5:  # Filter detections based on confidence score
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)  # Red box
            cv2.putText(frame, f'Object {int(cls)}: {prob:.2f}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model

# Create pipeline
pipeline = dai.Pipeline()

# Define the color camera (RGB)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Create output link to send RGB frames to the host
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Start data streams
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # Get the RGB frame from the OAK-D camera
        in_rgb = rgb_queue.get()  # Blocking call, will wait until a new data is available

        # Convert the frame into OpenCV format
        frame = in_rgb.getCvFrame()

        # Draw the boxes on the frame
        draw_boxes(frame)

        # Define the middle box (zone for detection)
        height, width, _ = frame.shape
        box_width = width // 3
        middle_box = (box_width, 0, 2 * box_width, height)  # (x_start, y_start, x_end, y_end)

        # Detect objects within the middle box
        detections = detect_objects_in_roi(frame, model, middle_box)

        # Draw bounding boxes around detected objects
        draw_detected_objects(frame, detections, middle_box)

        # Display the frame
        cv2.imshow("Obstacle Detection with OAK-D", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the OpenCV window
    cv2.destroyAllWindows()
