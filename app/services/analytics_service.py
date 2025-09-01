from typing import List, Dict

class AnalyticsService:
    def calculate_video_summary(self, detections: List[Dict], video_name: str) -> Dict:
        if not detections:
            return {}

        first_detection = detections[0]
        fps = first_detection.get('fps') or 30.0
        
        last_frame = max(d.get('frame_number', 0) for d in detections)
        video_duration_seconds = (last_frame + 1) / fps

        analysis = {}
        for det in detections:
            brand = det.get('brand_name')
            if not brand: continue
            if brand not in analysis:
                analysis[brand] = {"unique_frames": set()}
            analysis[brand]["unique_frames"].add(det.get('frame_number'))

        summary = []
        for brand, data in analysis.items():
            appearance_frames = len(data["unique_frames"])
            appearance_time_seconds = appearance_frames / fps
            screen_time_percentage = (appearance_time_seconds / video_duration_seconds) * 100 if video_duration_seconds > 0 else 0
            
            summary.append({
                "brand_name": brand,
                "appearance_time_seconds": round(appearance_time_seconds, 2),
                "screen_time_percentage": round(screen_time_percentage, 2)
            })
        
        return {
            "video_name": video_name,
            "video_duration_seconds": round(video_duration_seconds, 2),
            "analysis_summary": summary
        }