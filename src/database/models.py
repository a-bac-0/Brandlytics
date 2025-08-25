from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    duration = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")

class Brand(Base):
    __tablename__ = 'brands'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    color_r = Column(Integer, default=255)
    color_g = Column(Integer, default=255)  
    color_b = Column(Integer, default=255)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="brand")

class Detection(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    brand_id = Column(Integer, ForeignKey('brands.id'), nullable=False)
    frame_idx = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    cropped_image = Column(LargeBinary, nullable=True)  # Store cropped brand image
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video", back_populates="detections")
    brand = relationship("Brand", back_populates="detections")

class AnalysisReport(Base):
    __tablename__ = 'analysis_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    total_brands_detected = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    report_data = Column(Text, nullable=True)  # JSON string with detailed analysis
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video")