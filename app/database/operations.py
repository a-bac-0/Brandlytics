import os
import io
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

import numpy as np
import av  # PyAV para abrir videos con FFmpeg
from PIL import Image
from supabase import create_client, Client
from ultralytics import YOLO  # YOLOv8 oficial

# ByteTrack
from yolox.tracker.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Variables de entorno
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class DetectionPipeline:
    def __init__(self, tmp_dir: str = "tmp_frames"):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)
        self.model = YOLO(YOLO_MODEL_PATH)
        self.tracker = BYTETracker(frame_rate=30)  # ajustable según tu FPS

    async def process_source(
        self,
        source: Union[str, Path],
        filename: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        """Procesa un video o imagen"""
        filename = filename or f"video_{int(datetime.now().timestamp())}"
        detections_all = []

        # Imagen simple
        if isinstance(source, Path) and source.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            pil_img = Image.open(source)
            dets = await self._detect_frame(pil_img)
            await self._save_detections(dets, filename, frame_number=1, timestamp=0.0)
            return dets

        # Video
        container = await self._open_video(source)
        stream = container.streams.video[0]
        frame_count = 0

        async for frame in self._frame_generator(container, stream, max_frames):
            frame_count += 1
            img_np = frame.to_ndarray(format='rgb24')
            pil_img = Image.fromarray(img_np)

            # Detección YOLO
            dets = await self._detect_frame(pil_img)

            # Tracking con ByteTrack
            dets_tracked = self._track_detections(dets)

            # Subir crops y guardar detecciones
            for det in dets_tracked:
                crop_img = pil_img.crop(det['bbox'])
                crop_bytes = self._pil_to_bytes(crop_img)
                crop_url = await self._upload_crop(crop_bytes, f"{filename}_{frame_count}_{det['brand_name']}.jpg")
                det['crop_url'] = crop_url

            # Guardar en DB detections y detection_events
            await self._save_detections(dets_tracked, filename, frame_number=frame_count, timestamp=float(frame.time))

            detections_all.append({
                "frame_idx": frame_count,
                "timestamp": float(frame.time),
                "detections": dets_tracked
            })

        return detections_all

    async def _open_video(self, source: Union[str, Path]):
        return av.open(str(source))

    async def _frame_generator(self, container, stream, max_frames=None):
        for frame_idx, frame in enumerate(container.decode(stream)):
            if max_frames and frame_idx >= max_frames:
                break
            yield frame

    async def _detect_frame(self, pil_img: Image.Image) -> List[Dict]:
        img_np = np.array(pil_img)
        results = self.model(img_np)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "brand_name": self.model.names[int(cls)],
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf)
            })
        return detections

    def _track_detections(self, detections: List[Dict]) -> List[Dict]:
        """Aplica ByteTrack y devuelve detecciones con track_id"""
        # Convertimos a formato requerido por ByteTrack
        dets_array = np.array([
            [det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], det['confidence']]
            for det in detections
        ])
        tracked = self.tracker.update(dets_array, img_shape=None)  # img_shape opcional
        # Asignamos track_id a cada detección
        for det, track in zip(detections, tracked):
            det['track_id'] = track.track_id
        return detections

    def _pil_to_bytes(self, pil_img: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        buf.seek(0)
        return buf.read()

    async def _upload_crop(self, img_bytes: bytes, filename: str) -> str:
        bucket_name = "crops"
        res = supabase.storage.from_(bucket_name).upload(filename, img_bytes, {"content-type": "image/jpeg"})
        if res.get("error"):
            logger.error(f"Error subiendo crop {filename}: {res['error']}")
            return ""
        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)
        return public_url.get("publicUrl", "")

    async def _save_detections(self, detections: List[Dict], video_name: str, frame_number: int, timestamp: float):
        """Guarda detecciones en la tabla 'detections' y agrupa eventos en 'detection_events'"""
        for det in detections:
            # Guardar detección individual
            data = {
                "video_name": video_name,
                "frame_number": frame_number,
                "timestamp_seconds": timestamp,
                "brand_name": det['brand_name'],
                "confidence": det['confidence'],
                "bbox_x1": det['bbox'][0],
                "bbox_y1": det['bbox'][1],
                "bbox_x2": det['bbox'][2],
                "bbox_y2": det['bbox'][3],
                "image_crop_url": det.get('crop_url', None),
                "detection_type": "video",
                "track_id": det.get('track_id', None),
                "fps": 30  # opcional, si sabes el FPS del video
            }
            supabase.table("detections").insert(data).execute()

        # Actualizar detection_events
        await self._update_detection_events(video_name)

    async def _update_detection_events(self, video_name: str):
        """Llena automáticamente detection_events desde detections"""
        query = f"""
        INSERT INTO detection_events (
            video_name, brand_name, track_id,
            start_frame, end_frame, total_frames,
            min_confidence, max_confidence, avg_confidence,
            first_detected, last_detected
        )
        SELECT
            video_name, brand_name, track_id,
            MIN(frame_number) AS start_frame,
            MAX(frame_number) AS end_frame,
            COUNT(*) AS total_frames,
            MIN(confidence) AS min_confidence,
            MAX(confidence) AS max_confidence,
            AVG(confidence) AS avg_confidence,
            MIN(created_at) AS first_detected,
            MAX(created_at) AS last_detected
        FROM detections
        WHERE video_name = '{video_name}'
        GROUP BY video_name, brand_name, track_id
        """
        supabase.postgrest.rpc("execute_sql", {"sql": query}).execute()


# Ejemplo de uso
if __name__ == "__main__":
    processor = DetectionPipeline()

    async def main():
        # Imagen individual
        await processor.process_source(Path("example_image.jpg"))

        # Video
        await processor.process_source("https://www.example.com/video.mp4", max_frames=10)

    asyncio.run(main())
