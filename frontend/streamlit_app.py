import streamlit as st
import requests
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Brand Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.status_code == 200, response.json()
    except:
        return False, None

def get_supported_brands():
    """Get list of supported brands"""
    try:
        response = requests.get(f"{API_BASE_URL}/brands")
        if response.status_code == 200:
            return response.json()["brands"]
        return []
    except:
        return []

def detect_brands_in_image(image_file):
    """Send image to API for brand detection"""
    try:
        files = {"image_file": image_file}
        response = requests.post(f"{API_BASE_URL}/api/detection/process-image", files=files)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error detecting brands: {e}")
        return []

def detect_brands_in_video(video_file, save_to_db=True):
    """Send video to API for brand detection"""
    try:
        files = {"video_file": video_file}
        data = {"save_to_db": save_to_db}
        
        with st.spinner("Processing video... This may take a while."):
            response = requests.post(
                f"{API_BASE_URL}/api/detection/process-video",
                files=files, 
                data=data,
                timeout=300  # 5 minute timeout
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def draw_detections_on_image(image, detections):
    """Draw bounding boxes on image"""
    for detection in detections:
        bbox = detection["bbox"]
        brand_name = detection["brand_name"]
        confidence = detection["confidence"]
        
        # Convert coordinates to int
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{brand_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def create_brand_analysis_charts(brand_analysis, video_duration):
    """Create visualizations for brand analysis"""
    if not brand_analysis:
        return None, None
    
    # Prepare data
    brands = list(brand_analysis.keys())
    appearances = [brand_analysis[brand]["total_appearances"] for brand in brands]
    percentages = [brand_analysis[brand]["appearance_percentage"] for brand in brands]
    confidences = [brand_analysis[brand]["average_confidence"] for brand in brands]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Brand Appearances", "Screen Time Percentage", 
                       "Average Confidence", "Time vs Confidence"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Bar chart of appearances
    fig.add_trace(
        go.Bar(x=brands, y=appearances, name="Appearances", marker_color="lightblue"),
        row=1, col=1
    )
    
    # Pie chart of screen time percentages
    fig.add_trace(
        go.Pie(labels=brands, values=percentages, name="Screen Time %"),
        row=1, col=2
    )
    
    # Bar chart of average confidence
    fig.add_trace(
        go.Bar(x=brands, y=confidences, name="Avg Confidence", marker_color="lightgreen"),
        row=2, col=1
    )
    
    # Scatter plot of time vs confidence
    times = [brand_analysis[brand]["total_time_seconds"] for brand in brands]
    fig.add_trace(
        go.Scatter(
            x=times, y=confidences, mode='markers+text', text=brands,
            textposition="top center", name="Time vs Confidence",
            marker=dict(size=10, color="red")
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Brand Detection Analysis")
    
    return fig, pd.DataFrame({
        'Brand': brands,
        'Appearances': appearances,
        'Screen Time %': percentages,
        'Avg Confidence': confidences,
        'Total Time (s)': times
    })

def main():
    st.title("🎯 Brand Detection System")
    st.markdown("**Computer Vision powered brand detection in images and videos**")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Check API status
    api_status, api_info = check_api_status()
    
    if api_status:
        st.sidebar.success("✅ API Connected")
        st.sidebar.json(api_info)
        
        # Get supported brands
        brands = get_supported_brands()
        if brands:
            st.sidebar.subheader("🏷️ Supported Brands")
            for brand in brands:
                st.sidebar.write(f"• {brand['name']}")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.error("Cannot connect to the API. Please ensure the backend is running.")
        return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎬 Video Analysis", "📊 Analytics"])
    
    with tab1:
        st.header("Image Brand Detection")
        
        uploaded_image = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect brands
            if st.button("🔍 Detect Brands", key="detect_image"):
                uploaded_image.seek(0)  # Reset file pointer
                api_response = detect_brands_in_image(uploaded_image)
                detections_list = api_response.get("detections", []) if api_response else []

                if detections_list:
                    with col2:
                        st.subheader("Detections")
                        # Draw detections
                        annotated_image = draw_detections_on_image(image.copy(), detections_list)
                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    
                    # Show detection details
                    st.subheader("Detection Results")
                    for i, detection in enumerate(detections_list):
                        with st.expander(f"Detection {i+1}: {detection['brand_name']}"):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Brand", detection['brand_name'])
                            with col_b:
                                st.metric("Confidence", f"{detection['confidence']:.2f}")
                            with col_c:
                                st.metric("Class ID", detection['class_id'])
                            
                            st.write("**Bounding Box:**", detection['bbox'])
                else:
                    st.warning("No brands detected in this image.")
    
    with tab2:
        st.header("Video Brand Analysis")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        save_to_db = st.checkbox("Save analysis to database", value=True)
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            # Video info
            st.subheader("📹 Video Information")
            file_size = len(uploaded_video.read()) / (1024 * 1024)  # MB
            uploaded_video.seek(0)
            st.write(f"**File size:** {file_size:.2f} MB")
            st.write(f"**File name:** {uploaded_video.name}")
            
            if st.button("🎬 Analyze Video", key="analyze_video"):
                uploaded_video.seek(0)
                results = detect_brands_in_video(uploaded_video, save_to_db)
                
                if results:
                    st.success("✅ Video analysis completed!")
                    
                    # Video processing stats
                    st.subheader("📊 Processing Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Duration", 
                            f"{results['video_info']['duration']:.1f}s"
                        )
                    with col2:
                        st.metric(
                            "Frames Processed", 
                            results['processing_stats']['total_frames_processed']
                        )
                    with col3:
                        st.metric(
                            "Processing Time", 
                            f"{results['processing_stats']['processing_time']:.1f}s"
                        )
                    with col4:
                        st.metric(
                            "Avg Time/Frame", 
                            f"{results['processing_stats']['avg_time_per_frame']:.3f}s"
                        )
                    
                    # Brand analysis
                    if results['brand_analysis']:
                        st.subheader("🏷️ Brand Detection Results")
                        
                        # Create visualizations
                        fig, df = create_brand_analysis_charts(
                            results['brand_analysis'], 
                            results['video_info']['duration']
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Data table
                            st.subheader("📈 Detailed Results")
                            st.dataframe(df, use_container_width=True)
                        
                        # Individual brand details
                        st.subheader("Brand Details")
                        for brand_name, analysis in results['brand_analysis'].items():
                            with st.expander(f"📊 {brand_name.title()} Analysis"):
                                col_a, col_b, col_c = st.columns(3)
                                
                                with col_a:
                                    st.metric("Total Appearances", analysis['total_appearances'])
                                    st.metric("Screen Time", f"{analysis['total_time_seconds']:.1f}s")
                                
                                with col_b:
                                    st.metric("Screen Time %", f"{analysis['appearance_percentage']:.1f}%")
                                    st.metric("Avg Confidence", f"{analysis['average_confidence']:.2f}")
                                
                                with col_c:
                                    st.metric("Max Confidence", f"{analysis['max_confidence']:.2f}")
                                    
                                    # Progress bar for screen time percentage
                                    st.progress(analysis['appearance_percentage'] / 100)
                    else:
                        st.warning("No brands detected in this video.")
                    
                    # Download links
                    if results.get('output_video_path'):
                        st.subheader("📥 Downloads")
                        st.info("Annotated video available for download via API endpoint.")
    
    with tab3:
        st.header("📊 Analytics Dashboard")
        st.info("This section would show historical analytics from the database.")
        st.markdown("**Features to implement:**")
        st.markdown("- Historical detection trends")
        st.markdown("- Brand popularity over time") 
        st.markdown("- Video processing statistics")
        st.markdown("- Model performance metrics")

if __name__ == "__main__":
    main()