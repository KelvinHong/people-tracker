import torch
from detector.yolo import get_model
import cv2
import time

def livestream_detector():
    model = get_model()
    cam = cv2.VideoCapture(0)

    instruction = "Detector"
    cv2.namedWindow(instruction)

    while True:
        start = time.time()
        ret, frame = cam.read()
        h, w = frame.shape[:2]
        if not ret:
            print("failed to grab frame")
            break
        results = model(frame)
        result = results.render()[0]
        delta_t = time.time() - start
        fps = max(1, round(1/delta_t))

        # Post process
        num = results.xyxy[0].shape[0]
        result = cv2.putText(result, f"{fps} FPS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (255,0,0), 2)
        result = cv2.putText(result, f"{num} people", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (255,0,0), 2)
        cv2.imshow(instruction, result)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    livestream_detector()
