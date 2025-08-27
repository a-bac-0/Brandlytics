# 🎯 TRELLO BOARD: 4-Day Computer Vision Brand Detection Project

## 📋 Board Structure
- **Columns**: Day 1 → Day 2 → Day 3 → Day 4 → Done
- **Cards**: Coded as `LEVEL.ROLE-TASK_NUMBER:DEPENDENCIES`
- **Labels**: 🟢 Esencial | 🟡 Medio | 🟠 Avanzado | 🔴 Experto
- **Team**: 5 members with specialized roles

---

## 👥 TEAM ROLES
1. **LT** = Líder Técnico (Model Training & AI)
2. **ED** = Especialista en Datos (Dataset & Preprocessing)  
3. **BE** = Backend Developer (Database & API)
4. **FE** = Frontend Developer (UI & Demo)
5. **PM** = Project Manager (Documentation & QA)

---

## 📅 DAY 1: PROJECT SETUP & FOUNDATIONS

### 🟢.LT-01: Setup Development Environment
**Dependencies**: None
**Description**: Install and configure all ML libraries (PyTorch, OpenCV, Ultralytics YOLO, etc.)
**Tasks**:
- [ ] Create virtual environment
- [ ] Install requirements.txt
- [ ] Test YOLO inference on sample image
- [ ] Document environment setup
**Deliverable**: Working ML environment with test inference

### 🟢.ED-01: Initial Dataset Collection
**Dependencies**: None  
**Description**: Collect and organize initial brand logo images for training
**Tasks**:
- [ ] Select 2-3 brands for detection (Nike, Coca-Cola, Apple, etc.)
- [ ] Collect minimum 100 images per brand
- [ ] Organize in folders: `data/raw/brand_name/`
- [ ] Basic quality check (resolution, clarity)
**Deliverable**: Organized image dataset structure

### 🟢.BE-01: Basic Detection Pipeline
**Dependencies**: 🟢.LT-01
**Description**: Create minimal script to load image and run inference
**Tasks**:
- [ ] Script to load single image
- [ ] Run YOLO inference 
- [ ] Extract bounding boxes and labels
- [ ] Save results to JSON format
**Deliverable**: `detect_image.py` script working

### 🟢.FE-01: Basic Streamlit Interface
**Dependencies**: None
**Description**: Create simple web interface for image upload
**Tasks**:
- [ ] Setup Streamlit environment
- [ ] File uploader for images
- [ ] Display uploaded image
- [ ] Basic UI structure
**Deliverable**: Working Streamlit app for image upload

### 🟢.PM-01: Project Infrastructure
**Dependencies**: None
**Description**: Setup repository structure and project management
**Tasks**:
- [ ] Initialize Git repository with proper structure
- [ ] Create README.md with project overview
- [ ] Setup Trello/GitHub Projects board
- [ ] Define commit conventions and branching strategy
**Deliverable**: Complete project structure and documentation

---

## 📅 DAY 2: MODEL TRAINING & DATA PROCESSING

### 🟢.LT-02: Train Initial Model
**Dependencies**: 🟢.ED-02, 🟢.LT-01
**Description**: Train YOLO model on annotated dataset
**Tasks**:
- [ ] Convert annotations to YOLO format
- [ ] Configure training parameters (epochs, batch_size)
- [ ] Train model on single brand
- [ ] Validate on test set
- [ ] Save best model weights
**Deliverable**: Trained model file (.pt) with metrics report

### 🟢.ED-02: Dataset Annotation & Augmentation  
**Dependencies**: 🟢.ED-01
**Description**: Annotate images and apply data augmentation
**Tasks**:
- [ ] Install LabelImg or use Roboflow
- [ ] Annotate bounding boxes for all collected images
- [ ] Apply augmentation (flip, crop, brightness, rotation)
- [ ] Split dataset (train/val/test: 70/20/10)
- [ ] Export in YOLO format
**Deliverable**: Annotated dataset with augmentation

### 🟢.BE-02: Database Schema Design
**Dependencies**: 🟢.BE-01
**Description**: Design and implement database for storing detections
**Tasks**:
- [ ] Design PostgreSQL schema (videos, detections, brands)
- [ ] Create database tables
- [ ] Implement CRUD operations
- [ ] Test data insertion and retrieval
**Deliverable**: Working database with basic operations

### 🟡.FE-02: Detection Visualization
**Dependencies**: 🟢.FE-01, 🟢.BE-01
**Description**: Display detection results with bounding boxes
**Tasks**:
- [ ] Integrate with detection pipeline
- [ ] Draw bounding boxes on images
- [ ] Show brand labels and confidence scores
- [ ] Add results display panel
**Deliverable**: UI showing detection results visually

### 🟢.PM-02: Documentation & Code Review
**Dependencies**: All Day 1 tasks
**Description**: Review code quality and update documentation
**Tasks**:
- [ ] Review all commits and code quality
- [ ] Update README with current progress
- [ ] Document API endpoints and functions
- [ ] Update Trello board with progress
**Deliverable**: Updated documentation and progress report

---

## 📅 DAY 3: VIDEO PROCESSING & ADVANCED FEATURES

### 🟡.LT-03: Video Detection Pipeline
**Dependencies**: 🟢.LT-02, 🟡.BE-03
**Description**: Adapt model to process video files frame by frame
**Tasks**:
- [ ] Implement video frame extraction
- [ ] Process frames with trained model
- [ ] Calculate temporal statistics (duration, percentage)
- [ ] Handle different video formats
**Deliverable**: Video processing pipeline with temporal analysis

### 🟠.ED-03: Multi-brand Dataset
**Dependencies**: 🟢.ED-02
**Description**: Expand dataset to include multiple brands
**Tasks**:
- [ ] Collect images for 2nd and 3rd brands
- [ ] Annotate new brand images
- [ ] Balance dataset across all brands
- [ ] Create validation videos with multiple brands
**Deliverable**: Multi-brand dataset ready for training

### 🟡.BE-03: Video Processing Backend
**Dependencies**: 🟢.BE-02
**Description**: Handle video upload and processing workflow
**Tasks**:
- [ ] Video file upload handling
- [ ] Frame extraction and processing queue
- [ ] Store detection results with timestamps
- [ ] Calculate time-based metrics per brand
**Deliverable**: Backend API for video processing

### 🟡.FE-03: Video Interface & Metrics
**Dependencies**: 🟡.FE-02, 🟡.BE-03
**Description**: Video upload interface with results dashboard
**Tasks**:
- [ ] Video file uploader
- [ ] Processing status indicator
- [ ] Results dashboard with time metrics
- [ ] Brand appearance timeline visualization
**Deliverable**: Complete video processing interface

### 🟠.PM-03: Integration Testing
**Dependencies**: All Day 2 tasks
**Description**: Test full pipeline integration and document architecture
**Tasks**:
- [ ] Test end-to-end video processing workflow
- [ ] Document system architecture
- [ ] Prepare technical presentation outline
- [ ] Validate database integrity
**Deliverable**: System integration documentation

---

## 📅 DAY 4: PRODUCTION DEPLOYMENT & FINAL DEMO

### 🟠.LT-04: Multi-brand Model Training
**Dependencies**: 🟠.ED-03, 🟡.LT-03
**Description**: Train final model with all brands and optimize for production
**Tasks**:
- [ ] Retrain YOLO with complete multi-brand dataset
- [ ] Optimize model (pruning, quantization)
- [ ] Convert to production format (ONNX/TorchScript)
- [ ] Final model validation and metrics
**Deliverable**: Production-ready multi-brand model

### 🟠.ED-04: Model Evaluation & Dataset Report
**Dependencies**: 🟠.LT-04
**Description**: Generate comprehensive model evaluation report
**Tasks**:
- [ ] Test model on held-out validation videos
- [ ] Calculate precision, recall, F1 for each brand
- [ ] Generate confusion matrix
- [ ] Document dataset statistics and quality
**Deliverable**: Model evaluation report with metrics

### 🔴.BE-04: Cloud Deployment & API
**Dependencies**: 🟠.LT-04, 🟡.BE-03
**Description**: Deploy model to cloud with REST API
**Tasks**:
- [ ] Setup cloud environment (Heroku/AWS/GCP)
- [ ] Deploy model with FastAPI
- [ ] Implement async video processing
- [ ] Document API endpoints
**Deliverable**: Cloud-deployed API with documentation

### 🔴.FE-04: Production Frontend
**Dependencies**: 🔴.BE-04, 🟡.FE-03
**Description**: Connect frontend to cloud API and polish interface
**Tasks**:
- [ ] Connect to deployed API
- [ ] Add error handling and loading states
- [ ] Polish UI/UX for presentation
- [ ] Add brand comparison features
**Deliverable**: Production-ready web application

### 🔴.PM-04: Final Integration & Presentation
**Dependencies**: All previous tasks
**Description**: Final demo preparation and presentation
**Tasks**:
- [ ] Coordinate final integration testing
- [ ] Prepare technical presentation slides
- [ ] Record demo video as backup
- [ ] Final repository cleanup and README
- [ ] Rehearse live demonstration
**Deliverable**: Complete presentation and live demo

---

## 🔄 DEPENDENCY MATRIX

| Task | Depends On | Blocks |
|------|------------|---------|
| 🟢.LT-01 | None | 🟢.BE-01, 🟢.LT-02 |
| 🟢.ED-01 | None | 🟢.ED-02 |
| 🟢.BE-01 | 🟢.LT-01 | 🟢.BE-02, 🟡.FE-02 |
| 🟢.FE-01 | None | 🟡.FE-02 |
| 🟢.PM-01 | None | 🟢.PM-02 |
| 🟢.LT-02 | 🟢.ED-02, 🟢.LT-01 | 🟡.LT-03 |
| 🟢.ED-02 | 🟢.ED-01 | 🟢.LT-02, 🟠.ED-03 |
| 🟢.BE-02 | 🟢.BE-01 | 🟡.BE-03 |
| 🟡.FE-02 | 🟢.FE-01, 🟢.BE-01 | 🟡.FE-03 |
| 🟡.LT-03 | 🟢.LT-02, 🟡.BE-03 | 🟠.LT-04 |
| 🟠.ED-03 | 🟢.ED-02 | 🟠.LT-04 |
| 🟡.BE-03 | 🟢.BE-02 | 🟡.LT-03, 🟡.FE-03 |
| 🟡.FE-03 | 🟡.FE-02, 🟡.BE-03 | 🔴.FE-04 |
| 🟠.LT-04 | 🟠.ED-03, 🟡.LT-03 | 🟠.ED-04, 🔴.BE-04 |
| 🟠.ED-04 | 🟠.LT-04 | Final Demo |
| 🔴.BE-04 | 🟠.LT-04, 🟡.BE-03 | 🔴.FE-04 |
| 🔴.FE-04 | 🔴.BE-04, 🟡.FE-03 | Final Demo |
| 🔴.PM-04 | All tasks | Project Completion |

---

## 🚨 CRITICAL PATH & RISKS

### **Critical Path**: 
🟢.ED-01 → 🟢.ED-02 → 🟢.LT-02 → 🟡.LT-03 → 🟠.LT-04 → 🔴.BE-04

### **Risk Mitigation**:
- **Day 1**: If model setup fails, use pre-trained YOLO without retraining
- **Day 2**: If annotation takes too long, use smaller dataset (50 images per brand)  
- **Day 3**: If video processing is slow, focus on shorter clips (30-60 seconds)
- **Day 4**: If cloud deployment fails, demo locally with ngrok tunnel

### **Success Criteria**:
- ✅ Working brand detection in images (Level 🟢)
- ✅ Video processing with brand labels (Level 🟡) 
- ✅ Confidence scores and database storage (Level 🟠)
- ✅ Web interface and API deployment (Level 🔴)

---

## 📊 DAILY STANDUP FORMAT

**What I completed yesterday:**
**What I'm working on today:**  
**Blockers/Dependencies needed:**
**Estimated completion time:**

---

**🎯 Final Deliverables Checklist:**
- [ ] GitHub repository with clean code
- [ ] Live demo of web application  
- [ ] Technical presentation (15-20 slides)
- [ ] Trello board showing completed tasks
- [ ] Working model detecting multiple brands
- [ ] Database with stored detections
- [ ] API deployed to cloud
- [ ] Complete documentation and README