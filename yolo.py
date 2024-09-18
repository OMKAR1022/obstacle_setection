import cv2
import depthai as dai
from ultralytics import YOLO
from gtts import gTTS
import os


# Function to give voice command
def give_voice_command(command):
    tts = gTTS(command, lang='en')
    tts.save('command.mp3')
    os.system("mpg123 command.mp3")
    os.remove('command.mp3')


# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model


# Function to perform object detection using YOLOv8
def detect_objects(frame):
    results = model(frame)

    # Ensure results is a list and take the first item if needed
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    print("Detection Results:", result)  # Print the result for debugging

    # Extract bounding boxes
    boxes = result.boxes
    if boxes is None:
        print("No bounding boxes detected")
        return None, None, None

    # Extract confidence scores and class names
    if hasattr(result, 'probs') and result.probs is not None:
        scores = result.probs.cpu().numpy()
    else:
        print("No confidence scores detected")
        scores = None

    # Class names
    class_names = result.names

    # Convert boxes to numpy array if needed
    boxes = boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format

    return boxes, scores, class_names


# Function to check if objects are in different regions (left, middle, right)
def is_object_in_zone(boxes, scores, width, height, box_width, threshold=0.5):
    # Initialize flags for detected objects in zones
    objects_in_zones = {'left': [], 'middle': [], 'right': []}

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        box_xmin = int(xmin * width)
        box_xmax = int(xmax * width)
        box_ymin = int(ymin * height)
        box_ymax = int(ymax * height)

        # Check if the detection is for a person and the score is above the threshold
        if scores[i] > threshold:
            # Check if the box is in the left zone
            if box_xmax > 0 and box_xmin < box_width:
                objects_in_zones['left'].append((box_xmin, box_ymin, box_xmax, box_ymax))

            # Check if the box is in the middle zone
            if box_xmax > box_width and box_xmin < 2 * box_width:
                objects_in_zones['middle'].append((box_xmin, box_ymin, box_xmax, box_ymax))

            # Check if the box is in the right zone
            if box_xmax > 2 * box_width and box_xmin < width:
                objects_in_zones['right'].append((box_xmin, box_ymin, box_xmax, box_ymax))

    return objects_in_zones


# Function to draw boxes and give commands based on detected objects
def process_frame(frame, detections):
    height, width, _ = frame.shape

    # Divide screen into left, middle, and right regions
    box_width = width // 3

    # Extract detection information
    boxes, scores, _ = detections

    # Initialize person_boxes
    person_boxes = []

    # Check if scores are not None and if boxes are present
    if scores is not None and len(scores) > 0 and boxes is not None:
        # Filter detections for 'person' objects with a score above 0.5
        person_boxes = [boxes[i] for i in range(len(scores)) if scores[i] > 0.5]

    # Draw the left, middle, and right boxes on the frame
    cv2.rectangle(frame, (0, 0), (box_width, height), (255, 0, 0), 2)  # Left box (blue)
    cv2.rectangle(frame, (box_width, 0), (2 * box_width, height), (0, 255, 0), 2)  # Middle box (green)
    cv2.rectangle(frame, (2 * box_width, 0), (width, height), (0, 0, 255), 2)  # Right box (red)

    # Check if objects are in different regions
    objects_in_zones = is_object_in_zone(person_boxes, scores, width, height, box_width)

    # Draw red bounding boxes around detected persons in the respective zones
    for zone, boxes in objects_in_zones.items():
        color = (0, 0, 255)  # Red color for the bounding box
        for (xmin, ymin, xmax, ymax) in boxes:
            # Draw bounding box around detected person
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

    # Navigation logic based on detected objects
    object_in_left = bool(objects_in_zones['left'])
    object_in_middle = bool(objects_in_zones['middle'])
    object_in_right = bool(objects_in_zones['right'])

    if object_in_middle:
        # If an object is in the middle, check for overlap with left or right zones
        if not object_in_left:
            give_voice_command("Move Left")
        elif not object_in_right:
            give_voice_command("Move Right")
        else:
            give_voice_command("Stop, obstacle ahead")
    elif object_in_left and not object_in_middle:
        if not object_in_right:
            give_voice_command("Move Right")
    elif object_in_right and not object_in_middle:
        if not object_in_left:
            give_voice_command("Move Left")

    return frame


# Set up OAK-D pipeline
pipeline = dai.Pipeline()

# Define the color camera (RGB)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Output link for the RGB frame
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Start the pipeline and process frames
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # Get the RGB frame from the OAK-D camera
        in_rgb = rgb_queue.get()
        frame = in_rgb.getCvFrame()

        # Perform object detection on the frame
        detections = detect_objects(frame)

        # Process the frame, detect zones, and give voice commands
        processed_frame = process_frame(frame, detections)

        # Display the frame
        cv2.imshow("Obstacle Avoidance", processed_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
