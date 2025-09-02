from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import requests
from app.utils.image_processing import crop_and_upload_detection
from app.services.image_detection import ImageDetectionService
from app.api.schemas.schemas_detection import DetectionCreate
from uuid import uuid4

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

class VideoDetectionService:
    def __init__(self, image_service: ImageDetectionService):
        self.image_service = image_service
        self.model = image_service.model
        self.class_names = getattr(self.model, "names", {})

    def _get_video_bytes_from_url(self, url: str) -> tuple[bytes, str]:
        url_lower = url.lower()
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            if not yt_dlp:
                raise RuntimeError("Para URLs de YouTube, instala 'yt-dlp': pip install yt-dlp")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                ydl_opts = {
                    "format": "best[ext=mp4]/best",
                    "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
                    "quiet": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    file_path = ydl.prepare_filename(info)
                
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                filename = os.path.basename(file_path)
                return video_bytes, filename
        else:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            filename = os.path.basename(url.split("?")[0]) or "video_from_url.mp4"
            return response.content, filename

    def process_video_input(self, video_file_bytes: Optional[bytes], filename: Optional[str], video_url: Optional[str]) -> Dict:
        if video_url:
            video_file_bytes, filename = self._get_video_bytes_from_url(video_url)
        
        if not video_file_bytes:
            raise ValueError("No se proporcionaron datos de vídeo.")
        
        if not filename:
            filename = "video.mp4"

        return self._process_frames(
            video_bytes=video_file_bytes,
            filename=filename,
            frame_step=10,
            conf=0.25,
            iou=0.6,
            save_crops=True
        )

    def _process_frames(self, video_bytes: bytes, filename: str, frame_step: int, conf: float, iou: float, save_crops: bool) -> Dict:
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el vídeo.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        detections_to_save: List[DetectionCreate] = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    results = self.image_service.detect_brands(frame, conf=conf, iou=iou)
                    
                    for det in results:
                        crop_url = None
                        if save_crops:
                            crop_url = crop_and_upload_detection(frame, det, f"videos/{Path(filename).stem}")

                        detections_to_save.append(DetectionCreate(
                            video_name=filename,
                            frame_number=frame_idx,
                            timestamp_seconds=round(frame_idx / fps, 3),
                            brand_name=det["class_name"],
                            confidence=det["confidence"],
                            bbox_x1=det["box"][0], bbox_y1=det["box"][1],
                            bbox_x2=det["box"][2], bbox_y2=det["box"][3],
                            image_crop_url=crop_url,
                            fps=fps,
                            detection_type="video"
                        ))
                frame_idx += 1
        finally:
            cap.release()
            os.unlink(tmp_path)

        return {
            "filename": filename,
            "fps": fps,
            "total_frames": total_frames,
            "detections_to_save": detections_to_save
        }