from typing import List, Dict, Optional, Union
from app.config.video_analysis_config import VideoAnalysisConfig, load_preset_config
from app.database.operations import get_video_detections_byname
from app.services.video_detection import video_detection_service
from pathlib import Path
import json
from datetime import datetime

class AnalyticsService:
    def __init__(self):
        self.default_config = VideoAnalysisConfig()
    
    def calculate_video_summary(self, detections: List[Dict], video_name: str) -> Dict:
        if not detections:
            return {}

        first_detection = detections[0]
        fps = first_detection.get('fps') or 30.0
        
        last_frame = max(d.get('frame_number', 0) for d in detections)
        video_duration_seconds = (last_frame + 1) / fps

        analysis = {}
        timestamps_by_brand = {}
        
        for det in detections:
            brand = det.get('brand_name')
            frame = det.get('frame_number')
            timestamp = det.get('timestamp_seconds', frame / fps)
            confidence = det.get('confidence', 0.0)
            
            if not brand: continue
            
            if brand not in analysis:
                analysis[brand] = {
                    "unique_frames": set(),
                    "total_detections": 0,
                    "avg_confidence": 0,
                    "total_confidence": 0,
                }
                timestamps_by_brand[brand] = []
            
            analysis[brand]["unique_frames"].add(frame)
            analysis[brand]["total_detections"] += 1
            analysis[brand]["total_confidence"] += confidence
            timestamps_by_brand[brand].append(timestamp)

        summary = []
        for brand, data in analysis.items():
            appearance_frames = len(data["unique_frames"])
            appearance_time_seconds = appearance_frames / fps
            screen_time_percentage = (appearance_time_seconds / video_duration_seconds) * 100 if video_duration_seconds > 0 else 0
            avg_confidence = data["total_confidence"] / data["total_detections"] if data["total_detections"] > 0 else 0
            
            sorted_timestamps = sorted(timestamps_by_brand[brand])
            
            segments = []
            if sorted_timestamps:
                segment_start = sorted_timestamps[0]
                last_timestamp = sorted_timestamps[0]
                
                for ts in sorted_timestamps[1:]:
                    if ts - last_timestamp > 2.0:  
                        segments.append([segment_start, last_timestamp])
                        segment_start = ts
                    last_timestamp = ts
                
                segments.append([segment_start, last_timestamp])
            
            summary.append({
                "brand_name": brand,
                "appearance_time_seconds": round(appearance_time_seconds, 2),
                "screen_time_percentage": round(screen_time_percentage, 2),
                "total_detections": data["total_detections"],
                "avg_confidence": round(avg_confidence, 4),
                "appearance_timeline": [{"start": round(s, 2), "end": round(e, 2)} for s, e in segments]
            })
        
        summary.sort(key=lambda x: x["screen_time_percentage"], reverse=True)
        
        return {
            "video_name": video_name,
            "video_duration_seconds": round(video_duration_seconds, 2),
            "total_frames_analyzed": last_frame + 1,
            "unique_brands_detected": len(analysis),
            "total_detections": sum(d["total_detections"] for d in analysis.values()),
            "analysis_summary": summary,
            "generated_at": datetime.now().isoformat()
        }
    
    async def analyze_video_file(
        self, 
        video_bytes: bytes, 
        filename: str, 
        preset: str = "balanced", 
        save_to_db: bool = True
    ) -> Dict:
        try:
            config = load_preset_config(preset)
            processing_config = config.get_processing_config()
        except ValueError:
            processing_config = {"frame_step": 30, "conf": 0.25, "iou": 0.6}
        
        detection_results = await video_detection_service.process_video_input(
            video_file_bytes=video_bytes,
            filename=filename,
            video_url=None,
            save_to_db=save_to_db,
            frame_step=processing_config.get('sampling_intervals', [10, 20, 30])[1],  # Use medium setting
            conf=processing_config.get('confidence_threshold', 0.25),
            iou=0.6,
            save_crops=True
        )
        
        detections = [det for det in detection_results.get("detections", [])]
        
        detection_summary = detection_results.get("summary", {})
        enhanced_analytics = self.calculate_video_summary(detections, filename)
        
        result = {
            **enhanced_analytics,
            "performance_metrics": {
                "total_frames": detection_summary.get("total_frames", 0),
                "processed_frames": detection_summary.get("processed_frames", 0),
                "brands_found": detection_summary.get("brands_found", []),
                "brand_counts": detection_summary.get("brand_counts", {}),
            },
            "metadata": detection_results.get("metadata", {})
        }
        
        return result
    
    def get_video_analysis_from_db(self, video_name: str) -> Dict:
        detections = get_video_detections_byname(video_name)
        if not detections:
            return {"error": f"No detections found for video: {video_name}"}
            
        return self.calculate_video_summary(detections, video_name)
    
    def export_analysis(self, analysis: Dict, format: str = "json", output_path: Optional[str] = None) -> Union[str, Dict]:
        if format == "json":
            result = json.dumps(analysis, indent=2)
        elif format == "csv":
            headers = ["brand_name", "appearance_time_seconds", "screen_time_percentage"]
            rows = [",".join(headers)]
            
            for brand in analysis.get("analysis_summary", []):
                row = [
                    brand["brand_name"],
                    str(brand["appearance_time_seconds"]),
                    str(brand["screen_time_percentage"])
                ]
                rows.append(",".join(row))
                
            result = "\n".join(rows)
        else:
            return analysis
            
        if output_path:
            with open(output_path, "w") as f:
                f.write(result)
                
        return result


analytics_service = AnalyticsService()