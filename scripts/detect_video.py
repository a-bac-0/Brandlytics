import cv2
from ultralytics import YOLO
import time

def process_video(model_path, video_path, output_path):

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Para guardar el video
