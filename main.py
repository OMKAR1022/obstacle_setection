import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np

# Function to draw the boxes (left, middle, right)
def draw_boxes(frame):
    height, width, _ = frame.shape
    box_width = width // 3

    cv2.rectangle(frame, (0, 0), (box_width, height), (255, 0, 0), 2)  # Left Box (Blue)
    cv2.rectangle(frame, (box_width, 0), (2 * box_width, height), (0, 255, 0), 2)  # Middle Box (Green)
    cv2.rectangle(frame, (2 * box_width, 0), (width, height), (0, 0, 255), 2)  # Right Box (Red)

    cv2.putText(frame, 'Left Box (Move Right)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, 'Middle Box (Safe Area)', (box_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, 'Right Box (Move Left)', (2 * box_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Function to perform object detection using YOLOv8
def detect_objects_in_roi(frame, model, roi_box):
    x_start, y_start, x_end, y_end = roi_box
    roi = frame[y_start:y_end, x_start:x_end]
    results = model(roi)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                class_id = int(boxes.cls.cpu().numpy()[i]) if boxes.cls is not None else -1
                prob = boxes.conf.cpu().numpy()[i] if boxes.conf is not None else 0
                xmin, ymin, xmax, ymax = box
                detections.append([xmin, ymin, xmax, ymax, class_id, prob])
    return detections

# Function to get depth information for the detected objects
# Function to get depth information for the detected objects
def get_depth_info(depth_frame, xmin, ymin, xmax, ymax):
    # Get the center point of the bounding box to estimate depth
    x_center = int((xmin + xmax) / 2)
    y_center = int((ymin + ymax) / 2)

    # Ensure the coordinates are within the depth frame bounds
    h, w = depth_frame.shape  # height and width of the depth frame
    if x_center >= w or y_center >= h or x_center < 0 or y_center < 0:
        return None  # Return None if out of bounds

    # Extract depth information for the center point (in millimeters)
    depth_value = depth_frame[y_center, x_center]

    # Convert depth to meters
    depth_in_meters = depth_value / 1000.0
    return depth_in_meters


# Function to draw detected objects with depth information
# Function to draw detected objects with depth information
def draw_detected_objects(frame, depth_frame, detections, roi_box):
    x_start, y_start, x_end, y_end = roi_box
    for detection in detections:
        xmin, ymin, xmax, ymax, cls, prob = detection
        xmin += x_start
        xmax += x_start
        ymin += y_start
        ymax += y_start

        if prob > 0.5:
            class_name = model.names[cls]
            depth = get_depth_info(depth_frame, int(xmin), int(ymin), int(xmax), int(ymax))
            if depth is not None:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, f'{class_name}: {prob:.2f} ({depth:.2f}m)', (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # No valid depth information, draw without depth
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame, f'{class_name}: {prob:.2f}', (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Create pipeline
pipeline = dai.Pipeline()

# Define color camera (RGB)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Define stereo depth camera
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Set the camera resolutions
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Link the stereo depth cameras
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Create output streams
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")

# Link camera previews
cam_rgb.preview.link(xout_rgb.input)
stereo.depth.link(xout_depth.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # Get RGB and depth frames
        in_rgb = rgb_queue.get()
        in_depth = depth_queue.get()

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        # Draw the boxes
        draw_boxes(frame)

        # Define the boxes for detection
        height, width, _ = frame.shape
        box_width = width // 3
        left_box = (0, 0, box_width, height)
        middle_box = (box_width, 0, 2 * box_width, height)
        right_box = (2 * box_width, 0, width, height)

        # Detect objects within each region and calculate depth
        left_detections = detect_objects_in_roi(frame, model, left_box)
        middle_detections = detect_objects_in_roi(frame, model, middle_box)
        right_detections = detect_objects_in_roi(frame, model, right_box)

        # Draw bounding boxes with depth information
        draw_detected_objects(frame, depth_frame, left_detections, left_box)
        draw_detected_objects(frame, depth_frame, middle_detections, middle_box)
        draw_detected_objects(frame, depth_frame, right_detections, right_box)

        # Display the frame
        cv2.imshow("Navigation with Depth Sensing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
