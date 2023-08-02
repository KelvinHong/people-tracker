import torch
import cv2
import time
import numpy as np
import argparse
from detector.yolo import get_model
from tracker.utils import Tracks, xyxy2uvgh, uvgh2xyxy, \
            measure_and_assign

def plot_tracks(frame: np.ndarray, new_tracks_xyxy: np.ndarray, 
                tracks: Tracks, fps: int = None):
    # frame is an image from cv2 (i.e., BGR format)
    # new_tracks_xyxy is numpy array shape [N, 5], unnormalized bbox with ID
    # fps optional.
    # line_coef [a,b,c] is optional: Represent equation ax+by+c=0 in unit square. 
    h, w = frame.shape[:2]
    line_scale = max(1, int(min(h,w)*0.007))
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1, int(min(h,w)*0.001))
    # Plot spliting line
    if tracks.split_config is not None:
        points = split_config["points"]
        x1, y1 = points[0]
        x2, y2 = points[1]
        x1, y1 = int(w*x1), int(h*y1)
        x2, y2 = int(w*x2), int(h*y2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), line_scale)
    # Plot bbox
    for bbox in new_tracks_xyxy:
        id = bbox[4]
        bbox = bbox[:4].astype(int)
        # Draw bounding box
        frame = cv2.rectangle(frame, bbox[:2], bbox[2:], 
                              (255, 0, 0), thickness=line_scale)
        # Draw ID
        frame = cv2.putText(frame, str(int(id)), (bbox[0], bbox[1]+30), font_face, font_scale, (255, 255, 0), line_scale)
    # Plot other info
    infos = []
    if fps is not None:
        infos.append(f"FPS {fps}")
    if tracks.split_config is not None:
        infos.append(f"Total: {tracks.in_traffic - tracks.out_traffic}")
        infos.append(f"Traffic: {tracks.in_traffic}")
    base_x = int(0.02*w)
    base_y = int(0.02*h) + 20
    base_h = 35 
    for ind, info in enumerate(infos):
        frame = cv2.putText(frame, info, (base_x, base_y + ind * base_h), 
                            font_face, font_scale, (0, 255, 0), line_scale)
    
    return frame




def tracking(video_path: str = None, split_config: dict = None):
    # If video_path is None, use webcam.
    model = get_model()
    if video_path is None:
        # Use webcam
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture('data/video1.mp4')
    instruction = "Detector"
    cv2.namedWindow(instruction)

    tracks = Tracks(split_config = split_config)
    previous_time = time.time() - 0.01
    img_counter = 0
    while True:
        ret, frame = cam.read()
        now = time.time()
        delta_t = now - previous_time
        if not ret:
            print("Finished")
            break
        h, w = frame.shape[:2]
        results = model(frame)


        dets = torch.clone(results.xyxy[0][:, :4])
        ## normalize
        dets[:, [0,2]] /= w
        dets[:, [1,3]] /= h
        ## Transform to uvgh mode
        dets_uvgh = xyxy2uvgh(dets) # [N, 4]
        # Calculate IOU then assign
        tracks_uvgh, tracks_cov = tracks.retrieve() # [M, 5]
        pairing_dict = measure_and_assign(tracks_uvgh, tracks_cov, dets_uvgh)
        # Kalman filter on tracks
        tracks.step(dets_uvgh, pairing_dict, delta_t)
        # Draw results on frame
        fps = max(1, round(1/delta_t))
        new_tracks_xyxy, _ = tracks.retrieve(use_visible=True) # Currently is still uvgh
        if new_tracks_xyxy.size != 0:
            new_tracks_xyxy[:, :4] = uvgh2xyxy(new_tracks_xyxy[:, :4])
            # Unnormalize 
            # Format is now [N, 5]
            new_tracks_xyxy[:, [0,2]] *= w
            new_tracks_xyxy[:, [1,3]] *= h 
            new_tracks_xyxy = np.round(new_tracks_xyxy)
        # Plot on frame
        frame = plot_tracks(frame, new_tracks_xyxy, tracks, fps=fps)

        cv2.imshow(instruction, frame)
        # cv2.imwrite(f"output/im_{img_counter:05d}.png", frame)
        img_counter += 1
        # Update timestamp
        previous_time = now
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cam.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human Tracking Program')
    parser.add_argument('--xyxy', nargs=4,
                        help='[4 numbers] Two coordinates for splitting line', default = None,
                        type=float)
    parser.add_argument('--inside', nargs=2,
                        help='[2 numbers] One coordinate indicate inside of image', default = None,
                        type=float)
    parser.add_argument('--use-video', type=str, default=None, help="Supply path to a video. Use Webcam if not given.")
    args = parser.parse_args()
    if args.xyxy is None or args.inside is None:
        print("Not using splitting line.")
        split_config = None
    else:
        print("Using splitting line...")
        split_config = {
            "points": [
                # points should starts and ends at boundary
                args.xyxy[:2], # [x1, y1]
                args.xyxy[2:], # [x2, y2]
                # [0.8, 0.],
                # [0.8, 1.]
            ],
            "in": args.inside, # Members of {0, 0.5, 1} x {0, 0.5, 1} \ (0.5, 0.5), 8 possibilities.
        }
        print("SPLIT configuration: ")
        print(split_config)
    tracking(
            args.use_video, 
            split_config = split_config,
        )
