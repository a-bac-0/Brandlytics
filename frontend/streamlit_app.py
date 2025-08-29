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

def detect_brands_in_image(image_file=None, image_url=None):
    """Send image to API for brand detection using the correct endpoint"""
    try:
        # Usar el endpoint correcto
        url = f"{API_BASE_URL}/api/detection/process-image"
        
        if image_file:
            files = {"image_file": image_file}
            data = {}
            response = requests.post(url, files=files, data=data)
        elif image_url:
            data = {"image_url": image_url}
            response = requests.post(url, data=data)
        else:
            return None
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error detecting brands: {e}")
        return None

def detect_brands_in_video(video_file, save_to_db=True):
    """Send video to API for brand detection"""
    try:
        files = {"file": video_file}
        data = {"save_to_db": save_to_db}
        
        with st.spinner("Processing video... This may take a while."):
            response = requests.post(
                f"{API_BASE_URL}/detect/video", 
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
        st.header("📷 Image Brand Detection")
        
        # Agregar un poco de estilo con CSS personalizado
        st.markdown("""
        <style>
        .detection-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .metric-card {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-container {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 10px;
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Opciones de entrada con mejor diseño
        st.markdown("### 🎯 Método de Entrada")
        input_method = st.radio(
            "",
            ["📁 Subir archivo", "🌐 URL de imagen"],
            key="input_method",
            horizontal=True
        )
        
        st.divider()
        
        image = None
        detection_results = None
        
        if input_method == "📁 Subir archivo":
            uploaded_image = st.file_uploader(
                "Selecciona una imagen para analizar",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="image_uploader",
                help="Formatos soportados: JPG, JPEG, PNG, BMP"
            )
            
            if uploaded_image is not None:
                # Botón de detección prominente
                detect_button = st.button(
                    "🔍 Detectar Marcas", 
                    key="detect_image_file",
                    type="primary",
                    use_container_width=True
                )
                
                if detect_button:
                    uploaded_image.seek(0)
                    with st.spinner("🔄 Analizando imagen..."):
                        detection_results = detect_brands_in_image(image_file=uploaded_image)
                
                # Cargar imagen para mostrar
                uploaded_image.seek(0)
                image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        
        else:  # URL de imagen
            image_url = st.text_input(
                "🌐 Introduce la URL de la imagen:",
                placeholder="https://ejemplo.com/imagen.jpg",
                key="image_url_input",
                help="Introduce una URL válida de imagen"
            )
            
            if image_url:
                try:
                    # Botón de detección prominente
                    detect_button = st.button(
                        "🔍 Detectar Marcas", 
                        key="detect_image_url",
                        type="primary",
                        use_container_width=True
                    )
                    
                    if detect_button:
                        with st.spinner("🔄 Analizando imagen..."):
                            detection_results = detect_brands_in_image(image_url=image_url)
                    
                    # Cargar imagen para procesamiento local
                    response = requests.get(image_url)
                    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)
                        
                except Exception as e:
                    st.error(f"❌ Error al cargar la imagen desde URL: {e}")
        
        # Mostrar comparación de imágenes si tenemos tanto la imagen como los resultados
        if image is not None:
            st.divider()
            st.markdown("### 🖼️ Análisis Visual")
            
            # Crear dos columnas para comparación lado a lado
            col1, col2 = st.columns(2, gap="medium")
            
            with col1:
                st.markdown("#### 📷 Imagen Original")
                with st.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    # Redimensionar imagen para mejor visualización
                    height, width = image.shape[:2]
                    max_height = 400
                    if height > max_height:
                        scale = max_height / height
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        image_resized = cv2.resize(image, (new_width, new_height))
                    else:
                        image_resized = image
                    
                    st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎯 Detecciones")
                with st.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    
                    if detection_results:
                        # Convertir resultados al formato esperado
                        detections_formatted = []
                        for detection in detection_results.get("detections", []):
                            formatted_detection = {
                                "brand_name": detection["class_name"],
                                "confidence": detection["confidence"],
                                "bbox": detection["box"],
                                "class_id": detection["class_id"]
                            }
                            detections_formatted.append(formatted_detection)
                        
                        if detections_formatted:
                            # Usar la imagen redimensionada para las detecciones
                            annotated_image = draw_detections_on_image(image_resized.copy(), detections_formatted)
                            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                            
                            # Mostrar número de detecciones
                            st.success(f"✅ {len(detections_formatted)} marca(s) detectada(s)")
                        else:
                            # Mostrar imagen original si no hay detecciones
                            st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)
                            st.warning("⚠️ No se detectaron marcas")
                    else:
                        # Mostrar imagen original mientras se procesa
                        st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)
                        if not detection_results:
                            st.info("👆 Haz clic en 'Detectar Marcas' para analizar")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

        # Mostrar resultados detallados debajo de las imágenes
        if detection_results and image is not None:
            st.divider()
            st.markdown("### 📊 Resultados Detallados")
            
            # Información general en una tarjeta
            st.markdown('<div class="detection-container">', unsafe_allow_html=True)
            
            detections_formatted = []
            for detection in detection_results.get("detections", []):
                formatted_detection = {
                    "brand_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "bbox": detection["box"],
                    "class_id": detection["class_id"]
                }
                detections_formatted.append(formatted_detection)
            
            if detections_formatted:
                # Métricas generales
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🎯 Total Detecciones", 
                        len(detections_formatted),
                        help="Número total de marcas detectadas"
                    )
                
                with col2:
                    avg_confidence = sum(d["confidence"] for d in detections_formatted) / len(detections_formatted)
                    st.metric(
                        "📈 Confianza Promedio", 
                        f"{avg_confidence:.2f}",
                        help="Confianza promedio de todas las detecciones"
                    )
                
                with col3:
                    max_confidence = max(d["confidence"] for d in detections_formatted)
                    st.metric(
                        "⭐ Mejor Detección", 
                        f"{max_confidence:.2f}",
                        help="Mayor confianza entre todas las detecciones"
                    )
                
                # Estado de la base de datos
                st.info(f"💾 {detection_results.get('database_status', 'Procesamiento completado')}")
                
                # Detalles de cada detección
                st.markdown("#### 🔍 Detalle por Marca")
                
                for i, detection in enumerate(detections_formatted):
                    with st.expander(f"🏷️ {detection['brand_name']} - Detección #{i+1}", expanded=i==0):
                        # Crear columnas para la información
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        
                        with detail_col1:
                            st.markdown("**📊 Información Básica**")
                            st.write(f"**Marca:** {detection['brand_name']}")
                            st.write(f"**ID Clase:** {detection['class_id']}")
                        
                        with detail_col2:
                            st.markdown("**🎯 Métricas**")
                            st.write(f"**Confianza:** {detection['confidence']:.3f}")
                            # Barra de progreso para la confianza
                            st.progress(detection['confidence'])
                        
                        with detail_col3:
                            st.markdown("**📐 Coordenadas**")
                            bbox = detection['bbox']
                            st.write(f"**X1, Y1:** ({bbox[0]:.0f}, {bbox[1]:.0f})")
                            st.write(f"**X2, Y2:** ({bbox[2]:.0f}, {bbox[3]:.0f})")
                            st.write(f"**Tamaño:** {abs(bbox[2]-bbox[0]):.0f} × {abs(bbox[3]-bbox[1]):.0f}")
            
            st.markdown('</div>', unsafe_allow_html=True)

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