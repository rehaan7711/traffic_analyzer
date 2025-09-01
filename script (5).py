# Create the emergency vehicle detector module
emergency_detector_code = '''import cv2
import numpy as np
from ultralytics import YOLO
import logging
from datetime import datetime

class EmergencyDetector:
    """
    Emergency vehicle detection class for identifying ambulances, fire trucks, and police cars.
    Uses specialized detection models and visual/textual cues.
    """
    
    def __init__(self, confidence_threshold=0.4):
        """
        Initialize emergency vehicle detector.
        
        Args:
            confidence_threshold: Minimum confidence for emergency vehicle detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize main YOLO model for general detection
        self._initialize_models()
        
        # Emergency vehicle indicators
        self.emergency_indicators = {
            'ambulance': {
                'colors': [(255, 255, 255), (255, 0, 0)],  # White, Red
                'text_patterns': ['ambulance', 'emergency', 'medical', '108', '102'],
                'shape_features': 'rectangular_tall'
            },
            'fire_truck': {
                'colors': [(0, 0, 255), (255, 255, 0)],  # Red, Yellow
                'text_patterns': ['fire', 'rescue', '101'],
                'shape_features': 'large_rectangular'
            },
            'police_car': {
                'colors': [(255, 255, 255), (0, 0, 0), (0, 0, 255)],  # White, Black, Red
                'text_patterns': ['police', 'patrol', 'cop', '100'],
                'shape_features': 'standard_car'
            }
        }
        
        # Flashing light detection parameters
        self.light_detection_params = {
            'red_range': [(0, 50, 50), (10, 255, 255)],  # HSV range for red
            'blue_range': [(100, 50, 50), (130, 255, 255)],  # HSV range for blue
            'min_area': 50,
            'brightness_threshold': 200
        }
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self):
        """Initialize YOLO models for emergency vehicle detection."""
        try:
            # Main detection model
            self.model = YOLO('yolov8m.pt')
            
            # Try to load specialized emergency vehicle model if available
            try:
                self.emergency_model = YOLO('emergency_vehicles.pt')  # Custom trained model
                self.logger.info("Custom emergency vehicle model loaded")
            except:
                self.emergency_model = None
                self.logger.info("Using general model for emergency detection")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def detect_emergency_vehicles(self, frame, vehicle_detections=None):
        """
        Detect emergency vehicles in frame.
        
        Args:
            frame: Input image frame
            vehicle_detections: Pre-detected vehicles to analyze
            
        Returns:
            List of emergency vehicle detections
        """
        emergency_detections = []
        
        try:
            # If no vehicle detections provided, detect all vehicles first
            if vehicle_detections is None:
                vehicle_detections = self._detect_all_vehicles(frame)
            
            # Analyze each vehicle for emergency indicators
            for vehicle in vehicle_detections:
                emergency_score, emergency_type = self._analyze_vehicle_for_emergency(
                    frame, vehicle
                )
                
                if emergency_score > self.confidence_threshold:
                    emergency_detection = {
                        'bbox': vehicle['bbox'],
                        'type': emergency_type,
                        'confidence': emergency_score,
                        'original_detection': vehicle,
                        'timestamp': datetime.now().isoformat()
                    }
                    emergency_detections.append(emergency_detection)
            
            # Additional custom model detection if available
            if self.emergency_model is not None:
                custom_detections = self._detect_with_custom_model(frame)
                emergency_detections.extend(custom_detections)
            
        except Exception as e:
            self.logger.error(f"Error in emergency vehicle detection: {str(e)}")
        
        return emergency_detections
    
    def _detect_all_vehicles(self, frame):
        """Detect all vehicles in frame using YOLO."""
        detections = []
        
        results = self.model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for vehicle classes (car, bus, truck)
                    if class_id in [2, 3, 5, 7] and confidence > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': result.names[class_id]
                        }
                        detections.append(detection)
        
        return detections
    
    def _analyze_vehicle_for_emergency(self, frame, vehicle):
        """
        Analyze a vehicle detection for emergency indicators.
        
        Args:
            frame: Input frame
            vehicle: Vehicle detection dictionary
            
        Returns:
            Tuple of (emergency_score, emergency_type)
        """
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_roi = frame[y1:y2, x1:x2]
        
        if vehicle_roi.size == 0:
            return 0.0, 'unknown'
        
        scores = {}
        
        # Analyze for each emergency vehicle type
        for emergency_type, indicators in self.emergency_indicators.items():
            score = 0.0
            
            # Color analysis
            color_score = self._analyze_colors(vehicle_roi, indicators['colors'])
            score += color_score * 0.3
            
            # Flashing light detection
            light_score = self._detect_flashing_lights(vehicle_roi)
            score += light_score * 0.4
            
            # Shape/size analysis
            shape_score = self._analyze_shape(vehicle, indicators['shape_features'])
            score += shape_score * 0.2
            
            # Text pattern detection (if applicable)
            text_score = self._detect_text_patterns(vehicle_roi, indicators['text_patterns'])
            score += text_score * 0.1
            
            scores[emergency_type] = score
        
        # Return the highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            return best_score, best_type
        
        return 0.0, 'unknown'
    
    def _analyze_colors(self, roi, target_colors):
        """Analyze ROI for specific emergency vehicle colors."""
        if roi.size == 0:
            return 0.0
        
        score = 0.0
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        for target_color in target_colors:
            # Calculate color similarity
            color_mask = self._create_color_mask(hsv, target_color)
            color_ratio = np.sum(color_mask) / color_mask.size
            score += color_ratio
        
        return min(score, 1.0)
    
    def _create_color_mask(self, hsv_image, bgr_color):
        """Create mask for specific color in HSV space."""
        # Convert BGR to HSV
        bgr_array = np.uint8([[bgr_color]])
        hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
        
        # Define HSV range around target color
        tolerance = 20
        lower = np.array([max(0, hsv_color[0] - tolerance), 50, 50])
        upper = np.array([min(179, hsv_color[0] + tolerance), 255, 255])
        
        mask = cv2.inRange(hsv_image, lower, upper)
        return mask
    
    def _detect_flashing_lights(self, roi):
        """Detect flashing emergency lights in ROI."""
        if roi.size == 0:
            return 0.0
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        score = 0.0
        
        # Detect red lights
        red_mask = cv2.inRange(hsv, 
                              np.array(self.light_detection_params['red_range'][0]),
                              np.array(self.light_detection_params['red_range'][1]))
        
        # Detect blue lights  
        blue_mask = cv2.inRange(hsv,
                               np.array(self.light_detection_params['blue_range'][0]),
                               np.array(self.light_detection_params['blue_range'][1]))
        
        # Find bright spots
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_spots = cv2.threshold(gray, self.light_detection_params['brightness_threshold'], 
                                    255, cv2.THRESH_BINARY)[1]
        
        # Combine masks and check for significant light presence
        combined_lights = cv2.bitwise_or(red_mask, blue_mask)
        light_areas = cv2.bitwise_and(combined_lights, bright_spots)
        
        # Calculate score based on light presence
        light_ratio = np.sum(light_areas > 0) / light_areas.size
        score = min(light_ratio * 5, 1.0)  # Amplify small light areas
        
        return score
    
    def _analyze_shape(self, vehicle, shape_feature):
        """Analyze vehicle shape characteristics."""
        x1, y1, x2, y2 = vehicle['bbox']
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        area = width * height
        
        score = 0.0
        
        if shape_feature == 'rectangular_tall':
            # Ambulances tend to be taller
            if 1.2 <= aspect_ratio <= 2.0:
                score = 0.8
        
        elif shape_feature == 'large_rectangular':
            # Fire trucks are typically large and wide
            if aspect_ratio >= 1.5 and area > 15000:
                score = 0.9
        
        elif shape_feature == 'standard_car':
            # Police cars have standard car proportions
            if 1.5 <= aspect_ratio <= 2.2:
                score = 0.7
        
        return score
    
    def _detect_text_patterns(self, roi, patterns):
        """Detect text patterns indicative of emergency vehicles."""
        # This is a simplified implementation
        # In practice, you'd use OCR like Tesseract or EasyOCR
        
        score = 0.0
        
        # Look for high contrast areas that might contain text
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection to find text-like regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Simple heuristic: emergency vehicles often have more text/markings
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.05:  # Threshold for text presence
            score = 0.5
        
        return score
    
    def _detect_with_custom_model(self, frame):
        """Detect using custom emergency vehicle model if available."""
        detections = []
        
        if self.emergency_model is None:
            return detections
        
        try:
            results = self.emergency_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence > self.confidence_threshold:
                            class_id = int(box.cls[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'type': result.names[class_id],
                                'confidence': confidence,
                                'original_detection': None,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'custom_model'
                            }
                            detections.append(detection)
        
        except Exception as e:
            self.logger.error(f"Error with custom model detection: {str(e)}")
        
        return detections
    
    def visualize_emergency_detections(self, frame, emergency_detections):
        """
        Visualize emergency vehicle detections with special highlighting.
        
        Args:
            frame: Input frame
            emergency_detections: List of emergency detections
            
        Returns:
            Frame with visualized emergency detections
        """
        vis_frame = frame.copy()
        
        for detection in emergency_detections:
            x1, y1, x2, y2 = detection['bbox']
            emergency_type = detection['type']
            confidence = detection['confidence']
            
            # Special emergency highlighting - thick red border
            cv2.rectangle(vis_frame, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 4)
            
            # Inner border for type-specific color
            type_colors = {
                'ambulance': (255, 255, 255),  # White
                'fire_truck': (0, 0, 255),     # Red
                'police_car': (255, 0, 0)      # Blue
            }
            
            color = type_colors.get(emergency_type, (0, 255, 255))  # Yellow default
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Emergency label with flashing effect (simplified)
            label = f"EMERGENCY: {emergency_type.upper()}"
            label_bg_color = (0, 0, 255)  # Red background
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(vis_frame,
                         (x1, y1 - text_height - baseline - 10),
                         (x1 + text_width + 10, y1),
                         label_bg_color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add confidence score
            conf_label = f"{confidence:.2f}"
            cv2.putText(vis_frame, conf_label, (x2 - 60, y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def get_emergency_alert_data(self, emergency_detections):
        """
        Generate alert data for emergency detections.
        
        Args:
            emergency_detections: List of emergency detections
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for detection in emergency_detections:
            alert = {
                'alert_id': f"EMRG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'vehicle_type': detection['type'],
                'confidence': detection['confidence'],
                'location': {
                    'bbox': detection['bbox'],
                    'center': [
                        (detection['bbox'][0] + detection['bbox'][2]) // 2,
                        (detection['bbox'][1] + detection['bbox'][3]) // 2
                    ]
                },
                'timestamp': detection['timestamp'],
                'priority': self._get_priority_level(detection['type']),
                'recommended_action': self._get_recommended_action(detection['type'])
            }
            alerts.append(alert)
        
        return alerts
    
    def _get_priority_level(self, vehicle_type):
        """Get priority level for different emergency vehicles."""
        priorities = {
            'ambulance': 'HIGH',
            'fire_truck': 'CRITICAL',
            'police_car': 'MEDIUM'
        }
        return priorities.get(vehicle_type, 'MEDIUM')
    
    def _get_recommended_action(self, vehicle_type):
        """Get recommended action for different emergency vehicles."""
        actions = {
            'ambulance': 'Clear lane immediately - Medical emergency',
            'fire_truck': 'URGENT: Clear all lanes - Fire emergency',
            'police_car': 'Give way and reduce speed - Police operation'
        }
        return actions.get(vehicle_type, 'Give way to emergency vehicle')

# Example usage and testing
if __name__ == "__main__":
    # Test the emergency detector
    detector = EmergencyDetector()
    
    print("Emergency Vehicle Detector module loaded successfully!")
'''

with open('emergency_detector.py', 'w') as f:
    f.write(emergency_detector_code)

print("emergency_detector.py created successfully!")
print(f"File size: {len(emergency_detector_code)} characters")