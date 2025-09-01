import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path

class VehicleDetector:
    """
    Vehicle detection class using YOLOv8 for detecting various vehicle types.
    """

    def __init__(self, model_path='yolov8m.pt', confidence_threshold=0.5):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # Initialize YOLO model
        self._initialize_model()

        # Define vehicle classes (COCO dataset classes)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            1: 'bicycle'  # Including bicycle as a vehicle type
        }

        # Additional mappings for better classification
        self.class_mappings = {
            'car': 'car',
            'motorcycle': 'motorcycle',
            'bus': 'bus', 
            'truck': 'truck',
            'bicycle': 'bicycle',
            'motorbike': 'motorcycle',
            'auto': 'auto',  # For three-wheelers
            'rickshaw': 'auto'
        }

        self.logger = logging.getLogger(__name__)

    def _initialize_model(self):
        """Initialize the YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLOv8 model loaded: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame.

        Args:
            frame: Input image/frame

        Returns:
            List of detection dictionaries with bbox, class, confidence
        """
        detections = []

        try:
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)

            # Process results
            for result in results:
                boxes = result.boxes

                if boxes is not None:
                    for box in boxes:
                        # Get detection data
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Filter by confidence threshold
                        if confidence < self.confidence_threshold:
                            continue

                        # Check if it's a vehicle class
                        class_name = self._get_vehicle_class_name(class_id, result.names)
                        if class_name is None:
                            continue

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Create detection dictionary
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2 - x1) * (y2 - y1)
                        }

                        detections.append(detection)

        except Exception as e:
            self.logger.error(f"Error in vehicle detection: {str(e)}")

        return detections

    def _get_vehicle_class_name(self, class_id, class_names):
        """
        Get vehicle class name from class ID.

        Args:
            class_id: YOLO class ID
            class_names: Class names from model

        Returns:
            Vehicle class name or None if not a vehicle
        """
        # Check if it's a known vehicle class
        if class_id in self.vehicle_classes:
            return self.vehicle_classes[class_id]

        # Check class name from model
        if class_id in class_names:
            class_name = class_names[class_id].lower()

            # Map to our vehicle categories
            for key, value in self.class_mappings.items():
                if key in class_name:
                    return value

        return None

    def classify_vehicle_size(self, detection):
        """
        Classify vehicle by size category.

        Args:
            detection: Detection dictionary

        Returns:
            Size category: 'small', 'medium', 'large'
        """
        area = detection['area']

        # Define size thresholds (adjustable based on video resolution)
        if area < 2000:
            return 'small'  # motorcycle, bicycle
        elif area < 8000:
            return 'medium'  # car, auto
        else:
            return 'large'  # bus, truck

    def filter_detections_by_roi(self, detections, roi_polygon):
        """
        Filter detections to only include those within region of interest.

        Args:
            detections: List of detection dictionaries
            roi_polygon: Polygon defining region of interest

        Returns:
            Filtered list of detections
        """
        filtered_detections = []

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if center point is inside ROI
            if cv2.pointPolygonTest(roi_polygon, (center_x, center_y), False) >= 0:
                filtered_detections.append(detection)

        return filtered_detections

    def non_max_suppression(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for NMS

        Returns:
            List of filtered detections
        """
        if len(detections) == 0:
            return []

        # Extract boxes and scores
        boxes = []
        scores = []

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            boxes.append([x1, y1, x2, y2])
            scores.append(detection['confidence'])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, iou_threshold)

        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []

    def get_vehicle_statistics(self, detections):
        """
        Get statistics about detected vehicles.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary with vehicle statistics
        """
        stats = {
            'total_count': len(detections),
            'by_type': {},
            'by_size': {},
            'avg_confidence': 0.0,
            'detection_density': 0.0
        }

        if len(detections) == 0:
            return stats

        # Count by type
        for detection in detections:
            vehicle_type = detection['class_name']
            stats['by_type'][vehicle_type] = stats['by_type'].get(vehicle_type, 0) + 1

            # Count by size
            size = self.classify_vehicle_size(detection)
            stats['by_size'][size] = stats['by_size'].get(size, 0) + 1

        # Calculate average confidence
        confidences = [d['confidence'] for d in detections]
        stats['avg_confidence'] = np.mean(confidences)

        # Calculate detection density (detections per area)
        total_area = sum(d['area'] for d in detections)
        if total_area > 0:
            stats['detection_density'] = len(detections) / total_area * 100000  # Normalized

        return stats

    def visualize_detections(self, frame, detections, show_confidence=True):
        """
        Visualize detections on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores

        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Choose color based on vehicle type
            color = self._get_color_for_class(class_name)

            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = class_name
            if show_confidence:
                label += f" {confidence:.2f}"

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw text background
            cv2.rectangle(vis_frame, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)

            # Draw text
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_frame

    def _get_color_for_class(self, class_name):
        """Get color for different vehicle classes."""
        colors = {
            'car': (0, 255, 0),        # Green
            'bus': (255, 0, 0),        # Blue  
            'truck': (0, 0, 255),      # Red
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (255, 0, 255),   # Magenta
            'auto': (0, 255, 255),      # Yellow
        }
        return colors.get(class_name.lower(), (128, 128, 128))  # Gray default

    def save_detections_to_json(self, detections, output_path):
        """
        Save detections to JSON file.

        Args:
            detections: List of detection dictionaries  
            output_path: Output file path
        """
        import json
        from datetime import datetime

        # Prepare data for JSON serialization
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(detections),
            'detections': []
        }

        for i, detection in enumerate(detections):
            json_detection = {
                'id': i,
                'class_name': detection['class_name'],
                'confidence': float(detection['confidence']),
                'bbox': {
                    'x1': int(detection['bbox'][0]),
                    'y1': int(detection['bbox'][1]),
                    'x2': int(detection['bbox'][2]),
                    'y2': int(detection['bbox'][3])
                },
                'area': int(detection['area']),
                'size_category': self.classify_vehicle_size(detection)
            }
            json_data['detections'].append(json_detection)

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        self.logger.info(f"Detections saved to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Test the vehicle detector
    detector = VehicleDetector()

    # Test with a sample image (replace with actual image path)
    # test_image = cv2.imread("test_image.jpg")
    # detections = detector.detect_vehicles(test_image)
    # print(f"Detected {len(detections)} vehicles")

    print("Vehicle Detector module loaded successfully!")
