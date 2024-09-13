import cv2
import depthai as dai
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os

# Function to give voice command
def give_voice_command(command):
    tts = gTTS(command, lang='en')
    tts.save('command.mp3')
    os.system("mpg123 command.mp3")

# Load TensorFlow Object Detection model (assuming a pre-trainedec model for person dettion)
MODEL_PATH = "/Users/omkar/Downloads/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"  # Example: Path to pre-trained model
detect_fn = tf.saved_model.load(MODEL_PATH)

# Labels for the detection model (assuming person detection)
LABELS = {1: 'person'}  # You may need to adjust based on your model's label map

# Function to perform object detection using TensorFlow
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    # Extract boxes, scores, and classes
    boxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
    scores = detections['detection_scores'][0].numpy()  # Confidence scores
    classes = detections['detection_classes'][0].numpy().astype(int)  # Object classes
    return boxes, scores, classes

# Function to check if objects overlap in different regions (left, middle, right)
def is_object_in_zone(boxes, width, height, box_width, threshold=0.7):
    object_in_left = False
    object_in_middle = False
    object_in_right = False

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        box_xmin = int(xmin * width)
        box_xmax = int(xmax * width)
        box_ymin = int(ymin * height)
        box_ymax = int(ymax * height)

        # Check if the box is in the left zone
        if box_xmax <= box_width and box_xmin < box_width:
            object_in_left = True

        # Check if the box is in the middle zone
        if box_xmin >= box_width and box_xmax <= 2 * box_width:
            object_in_middle = True

        # Check if the box is in the right zone
        if box_xmin >= 2 * box_width and box_xmax <= width:
            object_in_right = True

    return object_in_left, object_in_middle, object_in_right

# Function to draw boxes and give commands based on detected objects
def process_frame(frame, detections):
    height, width, _ = frame.shape

    # Divide screen into left, middle, and right regions
    box_width = width // 3

    # Extract detection information
    boxes, scores, classes = detections

    # Filter detections for 'person' objects with a score above 0.5
    person_boxes = [boxes[i] for i in range(len(scores)) if scores[i] > 0.5 and classes[i] == 1]

    # Check if person-like objects are in the left, middle, or right box
    object_in_left, object_in_middle, object_in_right = is_object_in_zone(person_boxes, width, height, box_width)

    # Draw the left, middle, and right boxes on the frame
    cv2.rectangle(frame, (0, 0), (box_width, height), (255, 0, 0), 2)  # Left box (blue)
    cv2.rectangle(frame, (box_width, 0), (2 * box_width, height), (0, 255, 0), 2)  # Middle box (green)
    cv2.rectangle(frame, (2 * box_width, 0), (width, height), (0, 0, 255), 2)  # Right box (red)

    # Navigation logic based on detected objects
    if object_in_middle:
        # If an object is in the middle, suggest moving left or right based on available space
        if not object_in_left:
            give_voice_command("Move Left")
        elif not object_in_right:
            give_voice_command("Move Right")
        else:
            give_voice_command("Stop, obstacle ahead")
    elif object_in_left and object_in_middle:
        give_voice_command("Move Right")
    elif object_in_right and object_in_middle:
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
