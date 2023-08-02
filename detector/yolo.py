import torch
import os

def get_model():
    # Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False)
    model = torch.hub.load('ultralytics/yolov5', 'custom', './yolov5s_openvino_model/') 
    # model = torch.hub.load(os.getcwd(), 'custom', './yolov5s_openvino_model/',  source='local') 
    model.classes = [0]
    model.conf = 0.3
    model.iou = 0.4  # NMS IoU threshold
    # model.agnostic = False  # NMS class-agnostic
    # model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)