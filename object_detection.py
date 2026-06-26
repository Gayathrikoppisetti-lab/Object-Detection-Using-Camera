from ultralytics import YOLO
import cv2
import contextlib, os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False, stream=False)
    annotated_frame = results[0].plot()

    # Show output in fullscreen
    cv2.namedWindow("YOLOv8 Object Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLOv8 Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
