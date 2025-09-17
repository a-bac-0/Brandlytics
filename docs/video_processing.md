# Video Processing System Documentation

## Overview

The video processing system extends the existing image detection pipeline to handle video files with advanced tracking capabilities. It maintains the same architectural patterns and integrates seamlessly with the existing Supabase database schema.

## Architecture

### Components

1. **OptimizedVideoProcessor** (`app/utils/video_processor.py`)
   - Core video processing engine using PyAV/FFmpeg
   - YOLO model for brand detection
   - ByteTracker for object tracking
   - Frame sampling and performance optimization

2. **VideoDetectionService** (`app/services/video_detection.py`)
   - Service layer wrapping the video processor
   - Singleton pattern for resource efficiency
   - Batch processing capabilities

3. **Video API Routes** (`app/api/routes/video.py`)
   - RESTful endpoints for video processing
   - File upload handling
   - Background task management

4. **Detection Events** (`app/database/detection_events.py`)
   - Tracking analytics and aggregation
   - Detection event generation from individual detections

## Database Schema

### Tables Used

#### `detections` Table
```sql
CREATE TABLE public.detections (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  video_name text NOT NULL,
  frame_number integer NOT NULL,
  timestamp_seconds numeric,
  brand_name text NOT NULL,
  confidence numeric NOT NULL,
  bbox_x1 integer NOT NULL,
  bbox_y1 integer NOT NULL,
  bbox_x2 integer NOT NULL,
  bbox_y2 integer NOT NULL,
  image_crop_url text,
  created_at timestamp with time zone DEFAULT now(),
  track_id uuid,
  fps numeric,
  detection_type text NOT NULL DEFAULT 'video'::text
);
```

#### `detection_events` Table
```sql
CREATE TABLE public.detection_events (
  id integer NOT NULL DEFAULT nextval('detection_events_id_seq'::regclass),
  detection_id uuid,
  video_name text,
  brand_name text,
  track_id uuid,
  start_frame integer,
  end_frame integer,
  total_frames bigint,
  min_confidence numeric,
  max_confidence numeric,
  avg_confidence numeric,
  first_detected timestamp with time zone,
  last_detected timestamp with time zone
);
```

## API Endpoints

### Video Processing

#### Upload and Process Video
```http
POST /api/video/upload
Content-Type: multipart/form-data

Parameters:
- file: Video file (mp4, avi, mov, mkv)
- video_name: Optional name for the video
- save_to_db: Boolean (default: true)
- max_frames: Optional frame limit for testing
- generate_output_video: Boolean (default: false)
```

#### Process Existing Video
```http
POST /api/video/process
Content-Type: application/json

{
  "video_name": "video_001",
  "save_to_db": true,
  "max_frames": null,
  "generate_output_video": false,
  "fps_sample": 1
}
```

#### Batch Processing
```http
POST /api/video/batch
Content-Type: application/json

{
  "video_names": ["video_001", "video_002"],
  "save_to_db": true,
  "max_frames": null
}
```

### Analytics and Events

#### Get Video Analytics
```http
GET /api/video/analytics/{video_name}
```

#### Get Detection Events
```http
GET /api/video/events/{video_name}
```

#### Get Brand-Specific Events
```http
GET /api/video/events/brand/{brand_name}?video_name=optional_filter
```

#### Generate Detection Events
```http
POST /api/video/events/generate/{video_name}
```

### Utility Endpoints

#### Get Processor Statistics
```http
GET /api/video/stats
```

#### List Available Videos
```http
GET /api/video/list
```

#### Download Annotated Video
```http
GET /api/video/download/{video_name}
```

## Configuration

### Environment Variables

```env
# Video Processing
VIDEO_FPS_SAMPLE=1                    # Frames per second to sample
VIDEO_MAX_FRAMES=0                    # Max frames to process (0 = no limit)
VIDEO_UPLOAD_DIR=data/raw/videos/input
VIDEO_OUTPUT_DIR=data/raw/videos/output

# ByteTracker Configuration
TRACKER_FRAME_RATE=30
TRACKER_TRACK_THRESH=0.5
TRACKER_TRACK_BUFFER=30
TRACKER_MATCH_THRESH=0.8
```

## Usage Examples

### Python Service Usage

```python
from app.services.video_detection import video_detection_service

# Process single video
results = await video_detection_service.process_video_file(
    video_path="path/to/video.mp4",
    video_name="my_video",
    save_to_db=True,
    max_frames=100
)

# Process batch
results = await video_detection_service.process_video_batch(
    video_paths=["video1.mp4", "video2.mp4"],
    save_to_db=True
)
```

### API Usage with curl

```bash
# Upload and process video
curl -X POST "http://localhost:8000/api/video/upload" \
  -F "file=@video.mp4" \
  -F "video_name=test_video" \
  -F "save_to_db=true"

# Get analytics
curl "http://localhost:8000/api/video/analytics/test_video"

# Generate detection events
curl -X POST "http://localhost:8000/api/video/events/generate/test_video"
```

## Performance Considerations

### Video Processing
- **PyAV/FFmpeg**: Optimized video decoding
- **Frame Sampling**: Configurable FPS sampling to balance speed vs accuracy
- **Batch Processing**: Efficient handling of multiple videos
- **Background Tasks**: Detection events generation runs asynchronously

### Memory Management
- **Streaming Processing**: Videos processed frame-by-frame
- **Crop Uploads**: Immediate upload to Supabase storage
- **Cleanup**: Automatic resource cleanup after processing

### Database Optimization
- **Batch Inserts**: Detections saved in batches
- **Indexing**: Recommended indexes on video_name, brand_name, track_id
- **Event Aggregation**: Efficient grouping of detections into tracking events

## Error Handling

The system includes comprehensive error handling at multiple levels:

1. **File Validation**: Video format and existence checks
2. **Processing Errors**: Graceful handling of detection failures
3. **Database Errors**: Transaction rollback and error reporting
4. **API Errors**: Structured HTTP error responses

## Future Enhancements

1. **Real-time Processing**: WebSocket support for live video streams
2. **Advanced Analytics**: Temporal analysis and trend detection
3. **Model Management**: Multiple model support and A/B testing
4. **Distributed Processing**: Queue-based processing for large videos
5. **Video Thumbnails**: Generate preview images with detection highlights
