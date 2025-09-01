import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import json
from datetime import datetime
import logging
from pathlib import Path

from vehicle_detector import VehicleDetector
from emergency_detector import EmergencyDetector
from centroid_tracker import CentroidTracker
from traffic_density import TrafficDensityAnalyzer
from junction_detector import JunctionDetector

class TrafficAnalyzer:
    """
    Main Traffic Analyzer class that orchestrates all components
    for comprehensive traffic video analysis.
    """

    def __init__(self, model_path='yolov8m.pt', confidence_threshold=0.5, 
                 light_traffic_max=40, heavy_traffic_min=65):
        """
        Initialize the Traffic Analyzer with all required components.

        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Detection confidence threshold
            light_traffic_max: Maximum percentage for light traffic
            heavy_traffic_min: Minimum percentage for heavy traffic
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.light_traffic_max = light_traffic_max
        self.heavy_traffic_min = heavy_traffic_min

        # Initialize components
        self._initialize_components()

        # Analysis state
        self.reset_analysis()

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        """Initialize all analyzer components."""
        try:
            self.vehicle_detector = VehicleDetector(
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold
            )

            self.emergency_detector = EmergencyDetector(
                confidence_threshold=self.confidence_threshold
            )

            self.centroid_tracker = CentroidTracker(max_disappeared=30)

            self.density_analyzer = TrafficDensityAnalyzer(
                light_max=self.light_traffic_max,
                heavy_min=self.heavy_traffic_min
            )

            self.junction_detector = JunctionDetector()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def reset_analysis(self):
        """Reset analysis state for new video processing."""
        self.vehicle_counts = defaultdict(int)
        self.emergency_alerts = []
        self.frame_analysis = []
        self.total_frames_processed = 0
        self.junction_type = "Unknown"
        self.overall_density = "Unknown"

    def analyze_video(self, video_path, progress_callback=None, save_output=True):
        """
        Analyze a traffic video and return comprehensive results.

        Args:
            video_path: Path to input video file
            progress_callback: Optional callback for progress updates
            save_output: Whether to save processed video

        Returns:
            Dict containing analysis results
        """
        try:
            self.reset_analysis()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")

            # Initialize video writer if saving output
            out_writer = None
            if save_output:
                output_path = self._get_output_path(video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            density_history = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process frame
                processed_frame, frame_results = self._process_frame(
                    frame, frame_count, width, height
                )

                # Store frame analysis
                self.frame_analysis.append({
                    'frame_number': frame_count,
                    'vehicle_count': frame_results['vehicle_count'],
                    'emergency_count': frame_results['emergency_count'],
                    'density_percentage': frame_results['density_percentage']
                })

                density_history.append(frame_results['density_percentage'])

                # Write processed frame
                if out_writer is not None:
                    out_writer.write(processed_frame)

                # Update progress
                if progress_callback and frame_count % 10 == 0:
                    progress = int(20 + (frame_count / total_frames) * 60)
                    progress_callback.progress(progress)

                # Process every nth frame for performance (adjustable)
                if frame_count % 2 == 0:
                    continue

            # Clean up
            cap.release()
            if out_writer is not None:
                out_writer.release()

            self.total_frames_processed = frame_count

            # Analyze overall traffic density
            if density_history:
                avg_density = np.mean(density_history)
                self.overall_density = self.density_analyzer.classify_density(avg_density)

            # Detect junction type (analyze first few frames)
            self.junction_type = self._detect_junction_type(video_path)

            # Compile results
            results = self._compile_results()

            self.logger.info(f"Analysis complete: {frame_count} frames processed")
            return results

        except Exception as e:
            self.logger.error(f"Error analyzing video: {str(e)}")
            raise

    def _process_frame(self, frame, frame_number, width, height):
        """
        Process a single frame for vehicle detection and analysis.

        Args:
            frame: Input frame
            frame_number: Current frame number
            width: Frame width
            height: Frame height

        Returns:
            Tuple of (processed_frame, frame_results)
        """
        processed_frame = frame.copy()

        # Detect vehicles
        vehicle_detections = self.vehicle_detector.detect_vehicles(frame)

        # Track vehicles
        tracked_objects = self.centroid_tracker.update(vehicle_detections)

        # Detect emergency vehicles
        emergency_detections = self.emergency_detector.detect_emergency_vehicles(
            frame, vehicle_detections
        )

        # Calculate traffic density
        density_percentage = self.density_analyzer.calculate_density(
            frame, vehicle_detections
        )

        # Draw detections and tracking
        processed_frame = self._draw_detections(
            processed_frame, vehicle_detections, tracked_objects, emergency_detections
        )

        # Update counters
        vehicle_count = 0
        for detection in vehicle_detections:
            vehicle_type = detection['class_name']
            self.vehicle_counts[vehicle_type] += 1
            vehicle_count += 1

        # Handle emergency alerts
        emergency_count = len(emergency_detections)
        if emergency_count > 0:
            for emergency in emergency_detections:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'frame_number': frame_number,
                    'vehicle_type': emergency['type'],
                    'confidence': emergency['confidence'],
                    'location': emergency['bbox']
                }
                self.emergency_alerts.append(alert)

        # Add overlay information
        processed_frame = self._add_overlay_info(
            processed_frame, vehicle_count, emergency_count, density_percentage
        )

        frame_results = {
            'vehicle_count': vehicle_count,
            'emergency_count': emergency_count,
            'density_percentage': density_percentage
        }

        return processed_frame, frame_results

    def _draw_detections(self, frame, vehicles, tracked_objects, emergencies):
        """Draw detection boxes and labels on frame."""
        # Draw vehicle detections
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            class_name = vehicle['class_name']
            confidence = vehicle['confidence']

            # Color based on vehicle type
            color = self._get_vehicle_color(class_name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw emergency vehicle alerts (with special highlighting)
        for emergency in emergencies:
            x1, y1, x2, y2 = emergency['bbox']

            # Red flashing effect for emergency vehicles
            cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 4)
            cv2.putText(frame, f"EMERGENCY: {emergency['type']}", 
                       (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw tracking IDs
        for obj_id, centroid in tracked_objects.items():
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 0), -1)
            cv2.putText(frame, f"ID: {obj_id}", 
                       (int(centroid[0])-10, int(centroid[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return frame

    def _add_overlay_info(self, frame, vehicle_count, emergency_count, density):
        """Add overlay information to frame."""
        height, width = frame.shape[:2]

        # Info panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Add text information
        info_texts = [
            f"Vehicles Detected: {vehicle_count}",
            f"Emergency Vehicles: {emergency_count}",
            f"Traffic Density: {density:.1f}%",
            f"Junction Type: {self.junction_type}",
            f"Frame: {self.total_frames_processed}"
        ]

        for i, text in enumerate(info_texts):
            y_pos = 30 + i * 20
            color = (0, 255, 0) if emergency_count == 0 else (0, 0, 255)
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def _get_vehicle_color(self, vehicle_type):
        """Get color for different vehicle types."""
        colors = {
            'car': (0, 255, 0),        # Green
            'bus': (255, 0, 0),        # Blue
            'truck': (0, 0, 255),      # Red
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (255, 0, 255),   # Magenta
            'auto': (0, 255, 255),      # Yellow
        }
        return colors.get(vehicle_type.lower(), (128, 128, 128))  # Gray default

    def _detect_junction_type(self, video_path):
        """Detect junction type from video analysis."""
        try:
            junction_type = self.junction_detector.detect_junction_type(video_path)
            return junction_type
        except Exception as e:
            self.logger.warning(f"Could not detect junction type: {str(e)}")
            return "Unknown"

    def _get_output_path(self, input_path):
        """Generate output video path."""
        input_path = Path(input_path)
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"processed_{input_path.stem}_{timestamp}.mp4"
        return str(output_dir / output_name)

    def _compile_results(self):
        """Compile final analysis results."""
        total_vehicles = sum(self.vehicle_counts.values())

        results = {
            'summary': {
                'total_vehicles': total_vehicles,
                'emergency_vehicles': len(self.emergency_alerts),
                'junction_type': self.junction_type,
                'overall_traffic_density': self.overall_density,
                'frames_processed': self.total_frames_processed,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'vehicle_counts': dict(self.vehicle_counts),
            'emergency_alerts': self.emergency_alerts,
            'frame_analysis': self.frame_analysis,
            'traffic_density_classification': {
                'light': f"< {self.light_traffic_max}%",
                'medium': f"{self.light_traffic_max}% - {self.heavy_traffic_min}%",
                'heavy': f"> {self.heavy_traffic_min}%"
            }
        }

        return results

    def export_results(self, results, output_format='json'):
        """
        Export analysis results to file.

        Args:
            results: Analysis results dictionary
            output_format: Export format ('json', 'csv', 'excel')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)

        if output_format == 'json':
            output_path = output_dir / f"traffic_analysis_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

        elif output_format == 'csv':
            import pandas as pd
            output_path = output_dir / f"traffic_analysis_{timestamp}.csv"

            # Create summary DataFrame
            summary_df = pd.DataFrame([results['summary']])
            summary_df.to_csv(output_path, index=False)

        self.logger.info(f"Results exported to {output_path}")
        return str(output_path)

# Example usage and testing
if __name__ == "__main__":
    # This section is for testing the traffic analyzer
    analyzer = TrafficAnalyzer()

    # Test with sample video (replace with actual video path)
    # results = analyzer.analyze_video("sample_traffic_video.mp4")
    # print(json.dumps(results, indent=2))

    print("Traffic Analyzer module loaded successfully!")
