import cv2


# Function to draw the boxes (left, middle, right) with specific purposes
def draw_boxes(frame):
    height, width, _ = frame.shape

    # Define boundaries for the left, middle, and right boxes
    box_width = width // 3  # Each box will be one-third of the screen width

    # Draw left box (for detecting objects that could enter middle from the left)
    cv2.rectangle(frame, (0, 0), (box_width, height), (255, 0, 0), 2)  # Blue box for left

    # Draw middle box (person's walking path)
    cv2.rectangle(frame, (box_width, 0), (2 * box_width, height), (0, 255, 0), 2)  # Green box for middle

    # Draw right box (for detecting objects that could enter middle from the right)
    cv2.rectangle(frame, (2 * box_width, 0), (width, height), (0, 0, 255), 2)  # Red box for right

    # Display labels on each box
    cv2.putText(frame, 'Left Box (Move Right if Object Enters)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                2)
    cv2.putText(frame, 'Middle Box (Walking Area)', (box_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, 'Right Box (Move Left if Object Enters)', (2 * box_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)


# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Draw the boxes on the frame
    draw_boxes(frame)

    # Display the resulting frame
    cv2.imshow('Obstacle Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
