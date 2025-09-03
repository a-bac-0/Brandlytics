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
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="Brand Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'api_status' not in st.session_state:
    st.session_state.api_status = False
if 'last_url' not in st.session_state:
    st.session_state.last_url = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False

# API Configuration
API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        st.error(f"API connection error: {str(e)}")
        return False, None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_supported_brands():
    """Get list of supported brands"""
    try:
        response = requests.get(f"{API_BASE_URL}/brands", timeout=5)
        if response.status_code == 200:
            return response.json()["brands"]
        return []
    except Exception as e:
        st.warning(f"Could not fetch supported brands: {str(e)}")
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
                f"{API_BASE_URL}/api/video/upload", 
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

def detect_brands_in_video_url(video_url, video_name=None, save_to_db=True):
    """Send video URL to API for brand detection"""
    if not video_url or not video_url.strip():
        st.error("Please provide a valid video URL")
        return None
        
    # Prevent processing the same URL repeatedly
    if st.session_state.get('last_processed_url') == video_url and st.session_state.get('processing'):
        st.warning("Already processing this URL. Please wait...")
        return None
    
    try:
        st.session_state.processing = True
        st.session_state.last_processed_url = video_url
        
        # Prepare data for URL processing
        params = {
            "url": video_url,
            "save_to_db": save_to_db
        }
        if video_name:
            params["video_name"] = video_name
        
        # Check if API is available
        api_status, _ = check_api_status()
        if not api_status:
            st.error("API is not available. Please check if the backend is running.")
            return None
        
        with st.spinner("🔄 Downloading and processing video... This may take a while."):
            response = requests.post(
                f"{API_BASE_URL}/api/video/process-url", 
                params=params,
                timeout=600  # 10 minute timeout for URL processing
            )
        
        if response.status_code == 200:
            st.success("✅ Video processed successfully!")
            return response.json()
        elif response.status_code == 503:
            st.error("🚫 URL processing is temporarily unavailable. Please try uploading the video file directly.")
            return None
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.Timeout:
        st.error("⏱️ Request timed out. The video might be too large or the server is busy.")
        return None
    except requests.ConnectionError:
        st.error("🔌 Connection error. Please check if the backend server is running.")
        return None
    except Exception as e:
        st.error(f"❌ Error processing video from URL: {str(e)}")
        return None
    finally:
        st.session_state.processing = False

def draw_detections_on_image(image, detections, original_shape, resized_shape):
    """Draw bounding boxes on image"""
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape
    width_scale = resized_width / original_width
    height_scale = resized_height / original_height
    for detection in detections.get("detections", []):
        bbox = detection["box"]
        brand_name = detection["class_name"]
        confidence = detection["confidence"]
        
        # Convert coordinates to int
        x1_orig, y1_orig, x2_orig, y2_orig = map(int, bbox)
        x1 = int(x1_orig * width_scale)
        y1 = int(y1_orig * height_scale)
        x2 = int(x2_orig * width_scale)
        y2 = int(y2_orig * height_scale)
        
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
    # Clear any stale processing state on app start
    if 'app_initialized' not in st.session_state:
        st.session_state.processing = False
        st.session_state.last_processed_url = ""
        st.session_state.app_initialized = True
    
    st.title("🎯 Brand Detection System")
    st.markdown("**Computer Vision powered brand detection in images and videos**")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Check API status with better error handling
    try:
        api_status, api_info = check_api_status()
        st.session_state.api_status = api_status
    except Exception as e:
        st.sidebar.error(f"❌ API Check Failed: {str(e)}")
        api_status = False
        api_info = None
    
    if api_status:
        st.sidebar.success("✅ API Connected")
        if api_info:
            st.sidebar.json(api_info)
        
        # Get supported brands
        try:
            brands = get_supported_brands()
            if brands:
                st.sidebar.subheader("🏷️ Supported Brands")
                for brand in brands:
                    if isinstance(brand, dict) and 'name' in brand:
                        st.sidebar.write(f"• {brand['name']}")
                    else:
                        st.sidebar.write(f"• {brand}")
        except Exception as e:
            st.sidebar.warning(f"Could not load brands: {str(e)}")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.error("Cannot connect to the API. Please ensure the backend is running on http://localhost:8000")
        st.info("💡 Try refreshing the page or check if the backend server is started.")
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
            "Selecciona el método de entrada:",
            ["📁 Subir archivo", "🌐 URL de imagen"],
            key="input_method",
            horizontal=True,
            label_visibility="collapsed"
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
                # Cargar imagen para mostrar
                pil_image = Image.open(uploaded_image).convert("RGB")
                pil_image = ImageOps.exif_transpose(pil_image)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Mostrar imagen original y botón
                st.divider()
                st.markdown("### 🖼️ Análisis Visual")
                
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
                        
                        st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Botón de detección dentro del container de imagen original
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
                
                with col2:
                    st.markdown("#### 🎯 Detecciones")
                    with st.container():
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        
                        if detection_results:
                            # Convertir resultados al formato esperado
                            if detection_results.get("detections"):
                            
                                # Usar la imagen redimensionada para las detecciones
                                annotated_image = draw_detections_on_image(
                                    image_resized.copy(), 
                                    detection_results,
                                    original_shape=image.shape[:2],
                                    resized_shape=image_resized.shape[:2]
                                )
                                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                                
                                # Mostrar número de detecciones
                                st.success(f"✅ {len(detection_results['detections'])} marca(s) detectada(s)")
                            else:
                                # Mostrar imagen original si no hay detecciones
                                st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.warning("⚠️ No se detectaron marcas")
                        else:
                            # Mostrar imagen original mientras se procesa
                            st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.info("👆 Haz clic en 'Detectar Marcas' para analizar")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # URL de imagen
            image_url = st.text_input(
                "🌐 Introduce la URL de la imagen:",
                placeholder="https://ejemplo.com/imagen.jpg",
                key="image_url_input",
                help="Introduce una URL válida de imagen"
            )
            
            if image_url:
                try:
                    # Cargar imagen para procesamiento local
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()
                    pil_image = Image.open(response.raw).convert("RGB")
                    pil_image = ImageOps.exif_transpose(pil_image)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    # Mostrar imagen y botón
                    st.divider()
                    st.markdown("### 🖼️ Análisis Visual")
                    
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
                        
                            st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Botón de detección dentro del container de imagen original
                            detect_button = st.button(
                                "🔍 Detectar Marcas", 
                                key="detect_image_url",
                                type="primary",
                                use_container_width=True
                            )
                            
                            if detect_button:
                                with st.spinner("🔄 Analizando imagen..."):
                                    detection_results = detect_brands_in_image(image_url=image_url)
                    
                    with col2:
                        st.markdown("#### 🎯 Detecciones")
                        with st.container():
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            
                            if detection_results:
                                # Convertir resultados al formato esperado
                                if detection_results.get("detections"):
                                    annotated_image = draw_detections_on_image(
                                        image_resized.copy(), 
                                        detection_results,
                                        original_shape=image.shape[:2],
                                        resized_shape=image_resized.shape[:2]
                                    )
                                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                                    st.success(f"✅ {len(detection_results['detections'])} marca(s) detectada(s)")
                                
                                
                                else:
                                    # Mostrar imagen original si no hay detecciones
                                    st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                                    st.warning("⚠️ No se detectaron marcas")
                            else:
                                # Mostrar imagen original mientras se procesa
                                st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.info("👆 Haz clic en 'Detectar Marcas' para analizar")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Error al cargar la imagen desde URL: {e}")

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
        st.header("🎬 Video Brand Analysis")
        
        # Agregar el mismo estilo CSS
        st.markdown("""
        <style>
        .video-container {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 10px;
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Opciones de entrada para video
        st.markdown("### 🎯 Método de Entrada")
        video_input_method = st.radio(
            "Selecciona el método de entrada para video:",
            ["📁 Subir archivo de video", "🌐 URL de video"],
            key="video_input_method",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        video_file = None
        video_url = None
        results = None
        
        if video_input_method == "📁 Subir archivo de video":
            uploaded_video = st.file_uploader(
                "Selecciona un video para analizar",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="video_uploader",
                help="Formatos soportados: MP4, AVI, MOV, MKV"
            )
            
            if uploaded_video is not None:
                video_file = uploaded_video
                
                # Mostrar información del video
                st.divider()
                st.markdown("### 📹 Información del Video")
                
                col1, col2 = st.columns(2, gap="medium")
                
                with col1:
                    st.markdown("#### 📷 Vista Previa")
                    with st.container():
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        st.video(uploaded_video)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Información del archivo
                        file_size = len(uploaded_video.read()) / (1024 * 1024)  # MB
                        uploaded_video.seek(0)
                        st.write(f"**📁 Tamaño:** {file_size:.2f} MB")
                        st.write(f"**📝 Nombre:** {uploaded_video.name}")
                        
                        # Checkbox para guardar en BD
                        save_to_db = st.checkbox("💾 Guardar análisis en base de datos", value=True, key="save_video_file")
                        
                        # Botón de análisis
                        analyze_button = st.button(
                            "🎬 Analizar Video", 
                            key="analyze_video_file",
                            type="primary",
                            use_container_width=True
                        )
                        
                        if analyze_button:
                            uploaded_video.seek(0)
                            with st.spinner("🔄 Procesando video... Esto puede tomar un tiempo..."):
                                results = detect_brands_in_video(uploaded_video, save_to_db)
                
                with col2:
                    st.markdown("#### 🎯 Resultados del Análisis")
                    with st.container():
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        
                        if results:
                            st.success("✅ ¡Análisis de video completado!")
                            
                            # Mostrar estadísticas básicas
                            if results.get('video_info'):
                                st.write(f"**⏱️ Duración:** {results['video_info']['duration']:.1f}s")
                            
                            if results.get('processing_stats'):
                                st.write(f"**🎞️ Frames procesados:** {results['processing_stats']['total_frames_processed']}")
                                st.write(f"**⏱️ Tiempo de procesamiento:** {results['processing_stats']['processing_time']:.1f}s")
                            
                            # Mostrar marcas detectadas
                            if results.get('brand_analysis'):
                                st.markdown("**🏷️ Marcas detectadas:**")
                                for brand_name in results['brand_analysis'].keys():
                                    st.write(f"• {brand_name.title()}")
                            else:
                                st.warning("⚠️ No se detectaron marcas en este video")
                        else:
                            st.info("👆 Haz clic en 'Analizar Video' para procesar")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # URL de video
            st.markdown("### 🌐 Procesamiento desde URL")
            st.info("💡 Pega aquí la URL de tu video para procesarlo automáticamente")
            
            # Initialize URL session state
            if 'video_url_input_field' not in st.session_state:
                st.session_state.video_url_input_field = ""
            
            video_url_input = st.text_input(
                "Introduce la URL del video:",
                value=st.session_state.get('video_url_input_field', ''),
                placeholder="https://www.youtube.com/watch?v=ejemplo o https://ejemplo.com/video.mp4",
                key="video_url_input_field",
                help="Acepta URLs de YouTube, Vimeo, o enlaces directos a archivos de video",
                max_chars=2000,
                label_visibility="visible"
            )
            
            # Show processing status
            if st.session_state.get('processing', False):
                st.warning("🔄 Processing video... Please wait and do not refresh the page.")
                
            # Reset button to clear URL and state
            if st.button("🗑️ Clear URL", help="Clear the current URL and reset"):
                st.session_state.video_url_input_field = ""
                st.session_state.processing = False
                st.session_state.last_processed_url = ""
                st.rerun()
            
            if video_url_input and video_url_input.strip():
                try:
                    video_url = video_url_input.strip()
                    
                    # Validate URL format
                    if not (video_url.startswith('http://') or video_url.startswith('https://')):
                        st.error("❌ Please enter a valid URL starting with http:// or https://")
                        st.stop()
                    
                    # Mostrar información del video
                    st.divider()
                    st.markdown("### 📹 Información del Video")
                    
                    col1, col2 = st.columns(2, gap="medium")
                    
                    with col1:
                        st.markdown("#### 📷 Vista Previa")
                        with st.container():
                            st.markdown('<div class="video-container">', unsafe_allow_html=True)
                            try:
                                st.video(video_url)
                                st.success("✅ Video cargado correctamente")
                            except Exception as e:
                                st.error(f"❌ Error al cargar el video: {e}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Checkbox para guardar en BD
                            save_to_db = st.checkbox("💾 Guardar análisis en base de datos", value=True, key="save_video_url")
                            
                            # Botón de análisis
                            analyze_button = st.button(
                                "🎬 Analizar Video", 
                                key="analyze_video_url",
                                type="primary",
                                use_container_width=True
                            )
                            
                            if analyze_button:
                                with st.spinner("🔄 Procesando video... Esto puede tomar un tiempo..."):
                                    # Process video from URL
                                    video_results = detect_brands_in_video_url(
                                        video_url=video_url,
                                        save_to_db=save_to_db
                                    )
                                    
                                    if video_results:
                                        # Store results in session state for persistence
                                        st.session_state.video_results = video_results
                                        st.session_state.current_video_source = "url"
                                        st.session_state.current_video_url = video_url
                                        st.success("✅ ¡Video procesado exitosamente!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Error al procesar el video. Verifica que la URL sea válida.")
                    
                    with col2:
                        st.markdown("#### 🎯 Resultados del Análisis")
                        with st.container():
                            st.markdown('<div class="video-container">', unsafe_allow_html=True)
                            
                            # Check if we have video results to display
                            if hasattr(st.session_state, 'video_results') and st.session_state.video_results:
                                video_results = st.session_state.video_results
                                
                                # Display summary statistics
                                st.success("✅ Video procesado exitosamente")
                                
                                stat_col1, stat_col2 = st.columns(2)
                                with stat_col1:
                                    # Handle brands_detected which might be a list or number
                                    brands_detected = video_results.get('brands_detected', 0)
                                    if isinstance(brands_detected, list):
                                        brands_count = len(brands_detected)
                                    else:
                                        brands_count = brands_detected
                                    
                                    st.metric("🏷️ Marcas Detectadas", brands_count)
                                    st.metric("⏱️ Tiempo de Procesamiento", f"{video_results.get('processing_time', 0):.1f}s")
                                with stat_col2:
                                    st.metric("🎞️ Frames Procesados", video_results.get('total_frames_processed', 0))
                                    st.metric("� Total Detecciones", video_results.get('total_detections', 0))
                                
                                # Display brand statistics if available
                                if video_results.get('brand_statistics'):
                                    st.markdown("**🏷️ Marcas Encontradas:**")
                                    for brand, count in video_results['brand_statistics'].items():
                                        st.write(f"• {brand}: {count} detecciones")
                                
                                # Add button to clear results
                                if st.button("🗑️ Limpiar Resultados", key="clear_url_results"):
                                    if hasattr(st.session_state, 'video_results'):
                                        del st.session_state.video_results
                                    if hasattr(st.session_state, 'current_video_source'):
                                        del st.session_state.current_video_source
                                    if hasattr(st.session_state, 'current_video_url'):
                                        del st.session_state.current_video_url
                                    st.rerun()
                            else:
                                st.info("�👆 Haz clic en 'Analizar Video' para procesar")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"❌ Error al cargar el video desde URL: {e}")

        # Mostrar resultados detallados debajo
        if results:
            st.divider()
            st.markdown("### 📊 Análisis Detallado")
            
            # Información general en una tarjeta
            st.markdown('<div class="detection-container">', unsafe_allow_html=True)
            
            # Estadísticas de procesamiento
            st.markdown("#### 📈 Estadísticas de Procesamiento")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric(
                    "⏱️ Duración", 
                    f"{results['video_info']['duration']:.1f}s",
                    help="Duración total del video"
                )
            with stat_col2:
                st.metric(
                    "🎞️ Frames Procesados", 
                    results['processing_stats']['total_frames_processed'],
                    help="Número de frames analizados"
                )
            with stat_col3:
                st.metric(
                    "⏱️ Tiempo de Procesamiento", 
                    f"{results['processing_stats']['processing_time']:.1f}s",
                    help="Tiempo total de procesamiento"
                )
            with stat_col4:
                st.metric(
                    "⚡ Velocidad", 
                    f"{results['processing_stats']['avg_time_per_frame']:.3f}s/frame",
                    help="Tiempo promedio por frame"
                )
            
            # Análisis de marcas
            if results['brand_analysis']:
                st.markdown("#### 🏷️ Resultados de Detección de Marcas")
                
                # Crear visualizaciones
                fig, df = create_brand_analysis_charts(
                    results['brand_analysis'], 
                    results['video_info']['duration']
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de datos
                    st.markdown("#### 📈 Tabla de Resultados")
                    st.dataframe(df, use_container_width=True)
                
                # Detalles de cada marca
                st.markdown("#### 🔍 Detalle por Marca")
                for brand_name, analysis in results['brand_analysis'].items():
                    with st.expander(f"🏷️ {brand_name.title()} - Análisis Detallado", expanded=False):
                        detail_col_a, detail_col_b, detail_col_c = st.columns(3)
                        
                        with detail_col_a:
                            st.markdown("**📊 Estadísticas Básicas**")
                            st.metric("Apariciones Totales", analysis['total_appearances'])
                            st.metric("Tiempo en Pantalla", f"{analysis['total_time_seconds']:.1f}s")
                        
                        with detail_col_b:
                            st.markdown("**🎯 Métricas de Confianza**")
                            st.metric("Confianza Promedio", f"{analysis['average_confidence']:.2f}")
                            st.metric("Confianza Máxima", f"{analysis['max_confidence']:.2f}")
                        
                        with detail_col_c:
                            st.markdown("**📈 Porcentajes**")
                            st.metric("% Tiempo en Pantalla", f"{analysis['appearance_percentage']:.1f}%")
                            # Barra de progreso para tiempo en pantalla
                            st.progress(analysis['appearance_percentage'] / 100)
                        
                        # Información adicional si existe
                        if 'detections' in analysis:
                            st.markdown("**🎬 Capturas de Detecciones**")
                            st.info("🚧 Las capturas de frames específicos estarán disponibles en la próxima versión.")
            else:
                st.warning("⚠️ No se detectaron marcas en este video.")
            
            # Enlaces de descarga
            if results.get('output_video_path'):
                st.markdown("#### 📥 Descargas")
                st.info("📹 Video anotado disponible para descarga a través del endpoint de la API.")
                if st.button("� Obtener enlace de descarga"):
                    st.code(f"{API_BASE_URL}/download/video/{results.get('video_id', 'latest')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
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