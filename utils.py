import cv2
import numpy as np
import json
import pandas as pd
import os
import tempfile
from datetime import datetime
from pathlib import Path
import logging

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('traffic_analyzer.log')
        ]
    )

def create_directories():
    """Create necessary directories for the project."""
    directories = ['models', 'uploads', 'outputs', 'reports', 'static', 'templates']

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    return directories

def save_processed_video(frames, output_path, fps=30):
    """
    Save processed frames as video file.

    Args:
        frames: List of processed frames
        output_path: Output video file path
        fps: Frames per second

    Returns:
        Path to saved video file
    """
    if not frames:
        raise ValueError("No frames to save")

    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()
    return output_path

def save_video_from_bytesio(video_data, filename):
    """
    Save BytesIO video data to file for OpenCV processing.

    Args:
        video_data: BytesIO video data from Streamlit file uploader
        filename: Temporary filename to save

    Returns:
        Path to saved temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.read())
        return tmp_file.name

def load_video_info(video_path):
    """
    Load basic information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info

def create_report(results, report_type='json'):
    """
    Create formatted report from analysis results.

    Args:
        results: Analysis results dictionary
        report_type: Type of report ('json', 'html', 'markdown')

    Returns:
        Formatted report string
    """
    if report_type == 'json':
        return json.dumps(results, indent=2)

    elif report_type == 'html':
        return create_html_report(results)

    elif report_type == 'markdown':
        return create_markdown_report(results)

    else:
        raise ValueError(f"Unsupported report type: {report_type}")

def create_html_report(results):
    """Create HTML report from results."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ color: #2E86C1; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background: #F8F9FA; padding: 10px; margin: 5px 0; border-left: 4px solid #2E86C1; }}
            .emergency {{ background: #FADBD8; border-left-color: #E74C3C; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1 class="header">Traffic Analysis Report</h1>

        <div class="section">
            <h2>Summary</h2>
            <div class="metric">Total Vehicles: {results['summary']['total_vehicles']}</div>
            <div class="metric {'emergency' if results['summary']['emergency_vehicles'] > 0 else ''}">
                Emergency Vehicles: {results['summary']['emergency_vehicles']}
            </div>
            <div class="metric">Junction Type: {results['summary']['junction_type']}</div>
            <div class="metric">Traffic Density: {results['summary']['overall_traffic_density']}</div>
        </div>

        <div class="section">
            <h2>Vehicle Classification</h2>
            <table>
                <tr><th>Vehicle Type</th><th>Count</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in results['vehicle_counts'].items()])}
            </table>
        </div>

        <div class="section">
            <h2>Analysis Details</h2>
            <p><strong>Frames Processed:</strong> {results['summary']['frames_processed']}</p>
            <p><strong>Analysis Time:</strong> {results['summary']['analysis_timestamp']}</p>
        </div>
    </body>
    </html>
    """

    return html_template

def create_markdown_report(results):
    """Create Markdown report from results."""
    markdown = f"""# Traffic Analysis Report

## Summary
- **Total Vehicles:** {results['summary']['total_vehicles']}
- **Emergency Vehicles:** {results['summary']['emergency_vehicles']}
- **Junction Type:** {results['summary']['junction_type']}
- **Traffic Density:** {results['summary']['overall_traffic_density']}
- **Frames Processed:** {results['summary']['frames_processed']}

## Vehicle Classification

| Vehicle Type | Count |
|--------------|-------|
"""

    for vehicle_type, count in results['vehicle_counts'].items():
        markdown += f"| {vehicle_type} | {count} |\n"

    if results['emergency_alerts']:
        markdown += """
## Emergency Vehicle Alerts

| Timestamp | Vehicle Type | Confidence | Frame |
|-----------|--------------|------------|-------|
"""
        for alert in results['emergency_alerts']:
            markdown += f"| {alert['timestamp']} | {alert['vehicle_type']} | {alert['confidence']:.2f} | {alert['frame_number']} |\n"

    markdown += f"""
## Analysis Information
- **Analysis Timestamp:** {results['summary']['analysis_timestamp']}
- **Traffic Density Classification:**
  - Light: {results['traffic_density_classification']['light']}
  - Medium: {results['traffic_density_classification']['medium']}
  - Heavy: {results['traffic_density_classification']['heavy']}
"""

    return markdown

def validate_model_files():
    """
    Validate that required model files exist.

    Returns:
        Dictionary with model availability status
    """
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    required_models = {
        'yolov8n.pt': 'YOLOv8 Nano model',
        'yolov8s.pt': 'YOLOv8 Small model', 
        'yolov8m.pt': 'YOLOv8 Medium model',
        'yolov8l.pt': 'YOLOv8 Large model',
        'yolov8x.pt': 'YOLOv8 Extra Large model'
    }

    model_status = {}

    for model_file, description in required_models.items():
        model_path = models_dir / model_file
        model_status[model_file] = {
            'exists': model_path.exists(),
            'description': description,
            'path': str(model_path)
        }

    return model_status

def download_model_if_needed(model_name='yolov8m.pt'):
    """
    Download YOLOv8 model if not already present.

    Args:
        model_name: Name of the model to download

    Returns:
        Path to the model file
    """
    try:
        from ultralytics import YOLO

        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / model_name

        # If model doesn't exist, YOLO will download it
        model = YOLO(model_name)

        # Move downloaded model to our models directory if needed
        if not model_path.exists():
            import shutil
            # Find where ultralytics stored the model
            default_path = Path.home() / '.ultralytics' / 'models' / model_name
            if default_path.exists():
                shutil.copy2(default_path, model_path)

        return str(model_path)

    except Exception as e:
        logging.error(f"Error downloading model {model_name}: {str(e)}")
        return None

def extract_video_frames(video_path, max_frames=None, step=1):
    """
    Extract frames from video for processing.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)
        step: Frame step (1 = every frame, 2 = every other frame, etc.)

    Returns:
        List of extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            frames.append(frame.copy())
            extracted_count += 1

            if max_frames and extracted_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frames

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

def non_max_suppression(detections, iou_threshold=0.5, score_threshold=0.3):
    """
    Apply Non-Maximum Suppression to remove duplicate detections.

    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold

    Returns:
        Filtered list of detections
    """
    if not detections:
        return []

    # Filter by score threshold
    filtered_detections = [d for d in detections if d['confidence'] >= score_threshold]

    if not filtered_detections:
        return []

    # Sort by confidence score (highest first)
    sorted_detections = sorted(filtered_detections, key=lambda x: x['confidence'], reverse=True)

    # Apply NMS
    keep = []
    while sorted_detections:
        current = sorted_detections.pop(0)
        keep.append(current)

        # Remove detections with high IoU
        sorted_detections = [
            d for d in sorted_detections 
            if calculate_iou(current['bbox'], d['bbox']) < iou_threshold
        ]

    return keep

def create_video_thumbnail(video_path, timestamp=None, size=(320, 240)):
    """
    Create thumbnail image from video.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds to extract frame (None for middle frame)
        size: Thumbnail size (width, height)

    Returns:
        Thumbnail image as numpy array
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine frame to extract
    if timestamp is None:
        # Use middle frame
        target_frame = frame_count // 2
    else:
        target_frame = int(timestamp * fps)

    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read frame from video")

    # Resize to thumbnail size
    thumbnail = cv2.resize(frame, size)

    return thumbnail

def format_duration(seconds):
    """
    Format duration in seconds to human readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def export_detections_to_csv(detections, output_path):
    """
    Export detection results to CSV file.

    Args:
        detections: List of detection dictionaries
        output_path: Output CSV file path
    """
    if not detections:
        # Create empty CSV with headers
        df = pd.DataFrame(columns=['frame', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    else:
        # Convert detections to DataFrame
        data = []
        for i, detection in enumerate(detections):
            row = {
                'frame': i,
                'class_name': detection.get('class_name', 'unknown'),
                'confidence': detection.get('confidence', 0.0),
                'x1': detection['bbox'][0],
                'y1': detection['bbox'][1],
                'x2': detection['bbox'][2],
                'y2': detection['bbox'][3]
            }
            data.append(row)

        df = pd.DataFrame(data)

    df.to_csv(output_path, index=False)

def cleanup_temp_files(temp_dir=None):
    """
    Clean up temporary files created during processing.

    Args:
        temp_dir: Specific directory to clean (None for system temp)
    """
    import glob
    import tempfile

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Remove temporary video files
    temp_patterns = [
        os.path.join(temp_dir, "tmp*.mp4"),
        os.path.join(temp_dir, "temp_video_*.mp4"),
        os.path.join(temp_dir, "processed_*.mp4")
    ]

    cleaned_count = 0
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                cleaned_count += 1
            except Exception as e:
                logging.warning(f"Could not remove temp file {file_path}: {str(e)}")

    logging.info(f"Cleaned up {cleaned_count} temporary files")

def get_system_info():
    """
    Get system information for debugging and optimization.

    Returns:
        Dictionary with system information
    """
    import platform
    import psutil

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        cuda_available = False
        cuda_device_count = 0

    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'cuda_available': cuda_available,
        'cuda_devices': cuda_device_count
    }

    return info

# Configuration management
class Config:
    """Configuration management class."""

    DEFAULT_CONFIG = {
        'model': {
            'name': 'yolov8m.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.4
        },
        'tracking': {
            'max_disappeared': 30,
            'max_distance': 50
        },
        'density': {
            'light_max': 40,
            'heavy_min': 65,
            'method': 'hybrid'
        },
        'output': {
            'save_video': True,
            'save_reports': True,
            'video_fps': 30
        }
    }

    def __init__(self, config_path=None):
        self.config_path = config_path or 'config.json'
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file or use defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                return {**self.DEFAULT_CONFIG, **config}
            except Exception as e:
                logging.warning(f"Could not load config: {str(e)}. Using defaults.")

        return self.DEFAULT_CONFIG.copy()

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save config: {str(e)}")

    def get(self, key, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

# Example usage and testing
if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test directory creation
    created_dirs = create_directories()
    print(f"Created directories: {created_dirs}")

    # Test model validation
    model_status = validate_model_files()
    print(f"Model status: {model_status}")

    # Test configuration
    config = Config()
    print(f"Default confidence threshold: {config.get('model.confidence_threshold')}")

    print("Utils module loaded successfully!")
