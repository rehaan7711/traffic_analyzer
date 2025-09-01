# Traffic Analyzer Project - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (recommended: Python 3.9 or 3.10)
- pip (Python package installer)
- Git (for cloning repositories)
- 4GB+ RAM recommended
- GPU (optional, for faster processing)

### Installation Steps

1. **Clone or Download the Project**
```bash
# If using git
git clone https://github.com/your-username/traffic-analyzer-project.git
cd traffic-analyzer-project

# Or download and extract the ZIP file
```

2. **Create Virtual Environment** (Recommended)
```bash
# Using venv
python -m venv traffic_env
source traffic_env/bin/activate  # On Windows: traffic_env\Scripts\activate

# Or using conda
conda create -n traffic_env python=3.9
conda activate traffic_env
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Pre-trained Models**
The YOLOv8 models will be automatically downloaded on first use. To pre-download:
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # This will download the model
```

5. **Run the Application**

**Option A: Streamlit (Recommended for beginners)**
```bash
streamlit run app.py
```

**Option B: Flask (For API/web service)**
```bash
python flask_app.py
```

## 📁 Project Structure

```
traffic-analyzer-project/
├── app.py                  # Streamlit main application
├── flask_app.py           # Flask alternative application  
├── traffic_analyzer.py    # Core traffic analysis module
├── vehicle_detector.py    # YOLOv8 vehicle detection
├── emergency_detector.py  # Emergency vehicle detection
├── centroid_tracker.py    # Vehicle tracking algorithm
├── traffic_density.py     # Traffic density analysis
├── junction_detector.py   # Junction type detection
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── models/              # Pre-trained models directory
├── uploads/             # Video upload directory
├── outputs/             # Processed video outputs
├── reports/             # Generated reports
├── static/              # Static files for web UI
└── templates/           # HTML templates for Flask
```

## 🛠 Configuration

### Model Selection
Choose the appropriate YOLOv8 model based on your needs:

- **yolov8n.pt**: Fastest, lowest accuracy (~6MB)
- **yolov8s.pt**: Fast, good accuracy (~22MB)  
- **yolov8m.pt**: Balanced speed/accuracy (~52MB) - **Recommended**
- **yolov8l.pt**: Slower, high accuracy (~110MB)
- **yolov8x.pt**: Slowest, highest accuracy (~220MB)

### Hardware Requirements

**Minimum:**
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB free space
- GPU: Not required (CPU processing)

**Recommended:**
- CPU: Quad-core processor (Intel i5/AMD Ryzen 5 or better)
- RAM: 8GB+
- Storage: 5GB+ free space
- GPU: NVIDIA GPU with CUDA support (optional, for faster processing)

### GPU Setup (Optional)
For GPU acceleration with CUDA:

1. **Install CUDA Toolkit**
   - Download from NVIDIA website
   - Install CUDA 11.8 or compatible version

2. **Install PyTorch with CUDA**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify GPU Installation**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")
```

## 📹 Usage Examples

### 1. Using Streamlit App
```bash
streamlit run app.py
```
- Open browser to http://localhost:8501
- Upload traffic video (MP4, AVI, MOV)
- Configure detection parameters
- View real-time analysis results

### 2. Using Flask API
```bash
python flask_app.py
```
- Open browser to http://localhost:5000
- Web interface similar to Streamlit
- RESTful API endpoints available

### 3. Programmatic Usage
```python
from traffic_analyzer import TrafficAnalyzer

# Initialize analyzer
analyzer = TrafficAnalyzer(
    model_path='yolov8m.pt',
    confidence_threshold=0.5
)

# Analyze video
results = analyzer.analyze_video('path/to/video.mp4')

# Print results
print(f"Total vehicles: {results['summary']['total_vehicles']}")
print(f"Junction type: {results['summary']['junction_type']}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error: No module named 'ultralytics'**
   ```bash
   pip install ultralytics
   ```

2. **CUDA Out of Memory**
   - Use smaller model (yolov8n.pt or yolov8s.pt)
   - Process video in smaller batches
   - Reduce video resolution

3. **OpenCV Video Codec Issues**
   ```bash
   pip install opencv-python-headless
   # Or
   conda install opencv
   ```

4. **Slow Processing on CPU**
   - Use smaller model (yolov8n.pt)
   - Process every 2nd or 3rd frame
   - Reduce video resolution

5. **Streamlit Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Model Download Issues
If models don't download automatically:
```python
import torch
from ultralytics import YOLO

# Force download
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
for model_name in models:
    model = YOLO(model_name)
    print(f"Downloaded: {model_name}")
```

### Video Format Issues
Supported formats: MP4, AVI, MOV, MKV

Convert unsupported formats:
```bash
# Using ffmpeg
ffmpeg -i input_video.webm -c:v libx264 -c:a aac output_video.mp4
```

## 📊 Features Overview

### Core Features
- ✅ Vehicle detection and counting
- ✅ Vehicle classification (car, bus, truck, motorcycle, bicycle)
- ✅ Emergency vehicle detection (ambulance, fire truck, police)
- ✅ Traffic density analysis (light/medium/heavy)
- ✅ Junction type detection (2-way, 3-way, 4-way)
- ✅ Real-time processing with progress tracking
- ✅ Comprehensive reporting (JSON, CSV, Excel)

### Advanced Features
- ✅ Centroid tracking for vehicle movement
- ✅ Traffic flow analysis
- ✅ Day/night processing capability
- ✅ Configurable detection thresholds
- ✅ Emergency vehicle alerts
- ✅ Web-based interface (Streamlit & Flask)
- ✅ RESTful API endpoints

### Output Formats
- **JSON**: Detailed analysis results
- **CSV**: Tabular data for spreadsheet analysis
- **Excel**: Multi-sheet reports
- **Processed Video**: Video with detection overlays
- **HTML Report**: Web-viewable analysis report

## 🌐 Web Interface Features

### Streamlit App
- Drag-and-drop video upload
- Real-time parameter adjustment
- Interactive results visualization
- Downloadable reports
- Progress tracking

### Flask App
- Web form upload interface
- Background processing
- REST API endpoints
- Analysis dashboard
- Programmatic access

## 📚 API Documentation

### Flask API Endpoints

**POST /upload**
- Upload video for analysis
- Parameters: video file, model choice, thresholds
- Returns: analysis_id for tracking

**GET /status/{analysis_id}**
- Check analysis progress
- Returns: status, progress percentage, messages

**GET /results/{analysis_id}**
- Get analysis results
- Returns: complete analysis data

**POST /api/upload_and_analyze**
- Synchronous analysis for API clients
- Returns: immediate results

**GET /api/models**
- List available YOLO models
- Returns: model info and descriptions

### Example API Usage
```python
import requests

# Upload video for analysis
with open('traffic_video.mp4', 'rb') as f:
    response = requests.post('http://localhost:5000/upload', 
                           files={'video': f},
                           data={'model': 'yolov8m.pt'})

analysis_id = response.json()['analysis_id']

# Check status
status = requests.get(f'http://localhost:5000/status/{analysis_id}')
print(status.json())

# Get results when complete
results = requests.get(f'http://localhost:5000/results/{analysis_id}')
print(results.json())
```

## 🔒 Security Notes

- The application processes uploaded videos locally
- No data is sent to external servers
- Temporary files are cleaned up automatically
- Use in trusted environments only

## 📈 Performance Optimization

### For Better Speed:
1. Use GPU acceleration (NVIDIA CUDA)
2. Choose lighter models (yolov8n.pt or yolov8s.pt)
3. Process every nth frame (skip frames)
4. Reduce video resolution
5. Use appropriate confidence thresholds

### For Better Accuracy:
1. Use larger models (yolov8l.pt or yolov8x.pt)
2. Lower confidence thresholds
3. Process all frames
4. Use higher resolution videos
5. Ensure good lighting in videos

## 🤝 Contributing

1. Fork the project
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 framework
- **OpenCV** for computer vision functionality
- **Streamlit** for rapid web app development
- **Flask** for web framework
- **PyTorch** for deep learning backend

## 📞 Support

For issues and questions:
1. Check this setup guide
2. Review troubleshooting section
3. Check project issues on GitHub
4. Create new issue with details

---

**Happy Analyzing! 🚗📊**
