import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from supabase import create_client, Client

logger = logging.getLogger(__name__)


class DetectionRepository:
    """Supabase repository for detections, analysis, and storage uploads."""

    def __init__(self) -> None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not configured")
        self.client: Client = create_client(url, key)
        self.bucket = os.getenv("BUCKET_CROPS", "crops")

    # Storage
    def upload_crop(self, img_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> Optional[str]:
        try:
            res = self.client.storage.from_(self.bucket).upload(filename, img_bytes, {"content-type": content_type})
            # supabase-py returns dict with 'error' only in older versions; handle both
            if isinstance(res, dict) and res.get("error"):
                logger.error("Storage upload error: %s", res.get("error"))
                return None
            public = self.client.storage.from_(self.bucket).get_public_url(filename)
            if isinstance(public, dict):
                return public.get("publicUrl")
            # Newer SDK may return string directly
            return str(public)
        except Exception as e:
            logger.exception("Failed to upload crop: %s", e)
            return None

    # Detections
    def insert_detection(self, data: Dict[str, Any]) -> None:
        self.client.table("detections").insert(data).execute()

    def insert_detections(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        self.client.table("detections").insert(rows).execute()

    # Analysis
    def insert_analysis_report(
        self,
        video_name: str,
        total_brands_detected: int,
        total_detections: int,
        report_data: Dict[str, Any],
    ) -> None:
        payload = {
            "video_name": video_name,
            "total_brands_detected": total_brands_detected,
            "total_detections": total_detections,
            "report_data": report_data,
            "generated_at": datetime.utcnow().isoformat(),
        }
        self.client.table("analysis_reports").insert(payload).execute()

    def get_analysis_by_video_name(self, video_name: str) -> Optional[Dict[str, Any]]:
        try:
            det_resp = self.client.table("detections").select("*").eq("video_name", video_name).execute()
            detections = getattr(det_resp, "data", None) or det_resp.get("data", [])  # type: ignore[attr-defined]

            rep_resp = self.client.table("analysis_reports").select("*").eq("video_name", video_name).order("generated_at", desc=True).limit(1).execute()
            reports = getattr(rep_resp, "data", None) or rep_resp.get("data", [])  # type: ignore[attr-defined]
            report = reports[0] if reports else None

            if not detections and not report:
                return None

            result: Dict[str, Any] = {
                "video_name": video_name,
                "detections": detections,
            }
            if report:
                result["analysis_report"] = report
            return result
        except Exception as e:
            logger.exception("Failed to fetch analysis for %s: %s", video_name, e)
            return None

    def get_video_statistics(self) -> Dict[str, Any]:
        try:
            # Counts
            det_count = self.client.table("detections").select("id", count="exact").execute()
            total_detections = getattr(det_count, "count", None) or det_count.get("count", 0)  # type: ignore[attr-defined]

            rep_count = self.client.table("analysis_reports").select("id", count="exact").execute()
            total_reports = getattr(rep_count, "count", None) or rep_count.get("count", 0)  # type: ignore[attr-defined]

            # Brand popularity (requires a view or we fetch and aggregate client-side for PoC)
            # For PoC do a simple select and aggregate in Python (acceptable for small data)
            dets_resp = self.client.table("detections").select("brand_name").execute()
            dets = getattr(dets_resp, "data", None) or dets_resp.get("data", [])  # type: ignore[attr-defined]
            pop: Dict[str, int] = {}
            for d in dets:
                b = d.get("brand_name")
                if not b:
                    continue
                pop[b] = pop.get(b, 0) + 1

            popularity = [{"brand": k, "detections": v} for k, v in sorted(pop.items(), key=lambda x: -x[1])]

            return {
                "overview": {
                    "total_detections": total_detections,
                    "total_reports": total_reports,
                },
                "brand_popularity": popularity,
            }
        except Exception as e:
            logger.exception("Failed to compute statistics: %s", e)
            return {"overview": {"total_detections": 0, "total_reports": 0}, "brand_popularity": []}
