import cv2
import numpy as np
from collections import deque
import logging
from datetime import datetime

class TrafficDensityAnalyzer:
    """
    Traffic density analysis module for classifying traffic as light, medium, or heavy
    based on vehicle occupancy and movement patterns.
    """

    def __init__(self, light_max=40, heavy_min=65, history_size=30):
        """
        Initialize traffic density analyzer.

        Args:
            light_max: Maximum percentage for light traffic classification
            heavy_min: Minimum percentage for heavy traffic classification  
            history_size: Number of frames to keep for temporal analysis
        """
        self.light_max = light_max
        self.heavy_min = heavy_min
        self.history_size = history_size

        # Density history for temporal analysis
        self.density_history = deque(maxlen=history_size)
        self.vehicle_count_history = deque(maxlen=history_size)
        self.speed_history = deque(maxlen=history_size)

        # Frame analysis parameters
        self.roi_points = None  # Region of interest for density calculation
        self.grid_size = 50     # Grid cell size for occupancy analysis
        self.frame_count = 0

        self.logger = logging.getLogger(__name__)

    def set_roi(self, points):
        """
        Set region of interest for density analysis.

        Args:
            points: List of (x, y) points defining ROI polygon
        """
        self.roi_points = np.array(points, dtype=np.int32)

    def calculate_density(self, frame, detections, method='occupancy'):
        """
        Calculate traffic density for current frame.

        Args:
            frame: Input frame
            detections: List of vehicle detections
            method: Density calculation method ('occupancy', 'count', 'hybrid')

        Returns:
            Density percentage (0-100)
        """
        self.frame_count += 1

        if method == 'occupancy':
            density = self._calculate_occupancy_density(frame, detections)
        elif method == 'count':
            density = self._calculate_count_density(frame, detections)
        elif method == 'hybrid':
            density = self._calculate_hybrid_density(frame, detections)
        else:
            density = self._calculate_occupancy_density(frame, detections)

        # Update history
        self.density_history.append(density)
        self.vehicle_count_history.append(len(detections))

        # Calculate average speed if detections have speed info
        avg_speed = self._calculate_average_speed(detections)
        self.speed_history.append(avg_speed)

        return density

    def _calculate_occupancy_density(self, frame, detections):
        """Calculate density based on area occupancy."""
        height, width = frame.shape[:2]

        # Create binary mask for vehicle occupancy
        occupancy_mask = np.zeros((height, width), dtype=np.uint8)

        # Fill vehicle areas in mask
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(occupancy_mask, (x1, y1), (x2, y2), 255, -1)

        # Apply ROI if defined
        if self.roi_points is not None:
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [self.roi_points], 255)

            # Calculate occupancy within ROI
            roi_area = np.sum(roi_mask > 0)
            occupied_area = np.sum(cv2.bitwise_and(occupancy_mask, roi_mask) > 0)

            if roi_area > 0:
                density = (occupied_area / roi_area) * 100
            else:
                density = 0
        else:
            # Calculate occupancy for entire frame
            total_area = width * height
            occupied_area = np.sum(occupancy_mask > 0)
            density = (occupied_area / total_area) * 100

        return density

    def _calculate_count_density(self, frame, detections):
        """Calculate density based on vehicle count."""
        height, width = frame.shape[:2]

        # Define maximum expected vehicles per area
        frame_area = width * height
        area_per_vehicle = 5000  # Average pixels per vehicle
        max_vehicles = frame_area / area_per_vehicle

        # Apply ROI if defined
        if self.roi_points is not None:
            roi_area = cv2.contourArea(self.roi_points)
            max_vehicles = roi_area / area_per_vehicle

            # Count vehicles within ROI
            vehicles_in_roi = 0
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if cv2.pointPolygonTest(self.roi_points, (center_x, center_y), False) >= 0:
                    vehicles_in_roi += 1

            density = min((vehicles_in_roi / max_vehicles) * 100, 100) if max_vehicles > 0 else 0
        else:
            # Use all detections
            density = min((len(detections) / max_vehicles) * 100, 100) if max_vehicles > 0 else 0

        return density

    def _calculate_hybrid_density(self, frame, detections):
        """Calculate density using hybrid approach combining occupancy and count."""
        occupancy_density = self._calculate_occupancy_density(frame, detections)
        count_density = self._calculate_count_density(frame, detections)

        # Weight occupancy more heavily as it's generally more accurate
        hybrid_density = (occupancy_density * 0.7) + (count_density * 0.3)

        return hybrid_density

    def _calculate_average_speed(self, detections):
        """Calculate average speed from detections if available."""
        speeds = []

        for detection in detections:
            if 'speed' in detection:
                speeds.append(detection['speed'])

        return np.mean(speeds) if speeds else 0.0

    def classify_density(self, density_percentage):
        """
        Classify density percentage into categories.

        Args:
            density_percentage: Density percentage (0-100)

        Returns:
            Classification string: 'Light', 'Medium', or 'Heavy'
        """
        if density_percentage < self.light_max:
            return 'Light'
        elif density_percentage > self.heavy_min:
            return 'Heavy' 
        else:
            return 'Medium'

    def get_temporal_classification(self, window_size=10):
        """
        Get density classification based on recent history.

        Args:
            window_size: Number of recent frames to consider

        Returns:
            Classification based on temporal analysis
        """
        if len(self.density_history) < window_size:
            window_size = len(self.density_history)

        if window_size == 0:
            return 'Unknown'

        # Get recent density values
        recent_densities = list(self.density_history)[-window_size:]

        # Calculate weighted average (recent frames weighted more)
        weights = np.linspace(1, 2, window_size)  # More weight to recent frames
        weighted_avg = np.average(recent_densities, weights=weights)

        return self.classify_density(weighted_avg)

    def detect_congestion_patterns(self):
        """
        Detect congestion patterns and traffic flow characteristics.

        Returns:
            Dictionary with traffic pattern analysis
        """
        if len(self.density_history) < 5:
            return {'pattern': 'insufficient_data'}

        densities = np.array(list(self.density_history))
        vehicle_counts = np.array(list(self.vehicle_count_history))
        speeds = np.array(list(self.speed_history))

        # Analyze trends
        density_trend = np.polyfit(range(len(densities)), densities, 1)[0]
        count_trend = np.polyfit(range(len(vehicle_counts)), vehicle_counts, 1)[0]

        # Analyze variability
        density_std = np.std(densities)
        count_std = np.std(vehicle_counts)

        # Classify patterns
        patterns = {
            'density_trend': 'increasing' if density_trend > 0.5 else 'decreasing' if density_trend < -0.5 else 'stable',
            'count_trend': 'increasing' if count_trend > 0.1 else 'decreasing' if count_trend < -0.1 else 'stable',
            'variability': 'high' if density_std > 10 else 'low' if density_std < 3 else 'medium',
            'current_density': densities[-1] if len(densities) > 0 else 0,
            'average_density': np.mean(densities),
            'peak_density': np.max(densities),
            'min_density': np.min(densities),
            'congestion_score': self._calculate_congestion_score(densities, speeds)
        }

        # Determine overall pattern
        if patterns['density_trend'] == 'increasing' and patterns['variability'] == 'high':
            patterns['pattern'] = 'building_congestion'
        elif patterns['density_trend'] == 'decreasing' and patterns['average_density'] > self.heavy_min:
            patterns['pattern'] = 'clearing_congestion'  
        elif patterns['variability'] == 'low' and patterns['average_density'] > self.heavy_min:
            patterns['pattern'] = 'persistent_heavy'
        elif patterns['variability'] == 'low' and patterns['average_density'] < self.light_max:
            patterns['pattern'] = 'free_flow'
        else:
            patterns['pattern'] = 'variable_flow'

        return patterns

    def _calculate_congestion_score(self, densities, speeds):
        """Calculate a congestion score based on density and speed."""
        if len(densities) == 0:
            return 0

        # Normalize density (0-100 scale)
        density_score = np.mean(densities) / 100

        # Normalize speed (inverse relationship - lower speed = higher congestion)
        if len(speeds) > 0 and np.mean(speeds) > 0:
            speed_score = 1 - (np.mean(speeds) / 50)  # Assuming max normal speed ~50
            speed_score = max(0, min(1, speed_score))  # Clamp to 0-1
        else:
            speed_score = 0.5  # Default if no speed data

        # Combine scores (density weighted more heavily)
        congestion_score = (density_score * 0.7) + (speed_score * 0.3)

        return min(congestion_score * 100, 100)  # Convert to 0-100 scale

    def create_density_heatmap(self, frame, detections, grid_size=None):
        """
        Create a density heatmap visualization.

        Args:
            frame: Input frame
            detections: Vehicle detections
            grid_size: Grid cell size for heatmap

        Returns:
            Heatmap overlay image
        """
        if grid_size is None:
            grid_size = self.grid_size

        height, width = frame.shape[:2]

        # Create grid
        grid_rows = height // grid_size
        grid_cols = width // grid_size

        heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)

        # Count vehicles in each grid cell
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            grid_x = min(center_x // grid_size, grid_cols - 1)
            grid_y = min(center_y // grid_size, grid_rows - 1)

            heatmap[grid_y, grid_x] += 1

        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(heatmap, (width, height))

        # Convert to color heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )

        return heatmap_colored

    def visualize_density_info(self, frame, density_percentage):
        """
        Add density information overlay to frame.

        Args:
            frame: Input frame
            density_percentage: Current density percentage

        Returns:
            Frame with density overlay
        """
        output_frame = frame.copy()
        height, width = output_frame.shape[:2]

        # Density classification
        classification = self.classify_density(density_percentage)

        # Color based on density level
        colors = {
            'Light': (0, 255, 0),    # Green
            'Medium': (0, 255, 255), # Yellow  
            'Heavy': (0, 0, 255)     # Red
        }

        color = colors.get(classification, (128, 128, 128))

        # Draw density bar
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 20

        # Background bar
        cv2.rectangle(output_frame, 
                     (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (0, 0, 0), -1)

        # Density fill
        fill_width = int((density_percentage / 100) * bar_width)
        cv2.rectangle(output_frame,
                     (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)

        # Text labels
        cv2.putText(output_frame, f"Density: {density_percentage:.1f}%",
                   (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(output_frame, f"Status: {classification}",
                   (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add temporal information if available
        if len(self.density_history) > 1:
            temporal_class = self.get_temporal_classification()
            cv2.putText(output_frame, f"Trend: {temporal_class}",
                       (bar_x, bar_y + bar_height + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_frame

    def export_density_analysis(self, output_path):
        """
        Export density analysis to JSON file.

        Args:
            output_path: Output file path
        """
        import json

        # Get traffic patterns
        patterns = self.detect_congestion_patterns()

        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'light_max': self.light_max,
                'heavy_min': self.heavy_min,
                'history_size': self.history_size
            },
            'current_status': {
                'density_percentage': list(self.density_history)[-1] if self.density_history else 0,
                'classification': self.get_temporal_classification(),
                'vehicle_count': list(self.vehicle_count_history)[-1] if self.vehicle_count_history else 0,
                'frame_count': self.frame_count
            },
            'historical_data': {
                'density_history': list(self.density_history),
                'vehicle_count_history': list(self.vehicle_count_history), 
                'speed_history': list(self.speed_history)
            },
            'traffic_patterns': patterns,
            'statistics': {
                'avg_density': np.mean(list(self.density_history)) if self.density_history else 0,
                'max_density': np.max(list(self.density_history)) if self.density_history else 0,
                'min_density': np.min(list(self.density_history)) if self.density_history else 0,
                'density_std': np.std(list(self.density_history)) if self.density_history else 0
            }
        }

        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        self.logger.info(f"Density analysis exported to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Test the traffic density analyzer
    analyzer = TrafficDensityAnalyzer()

    # Simulate some detections
    test_detections = [
        {'bbox': [100, 100, 150, 150]},
        {'bbox': [200, 200, 250, 250]},
        {'bbox': [300, 300, 350, 350]},
    ]

    # Test frame (black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Calculate density
    density = analyzer.calculate_density(test_frame, test_detections)
    classification = analyzer.classify_density(density)

    print(f"Density: {density:.2f}%")
    print(f"Classification: {classification}")

    print("Traffic Density Analyzer module loaded successfully!")
