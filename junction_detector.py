import cv2
import numpy as np
from collections import Counter
import logging

class JunctionDetector:
    """
    Junction detection module for identifying intersection types (2-way, 3-way, 4-way)
    based on road structure analysis using computer vision techniques.
    """

    def __init__(self):
        """Initialize the junction detector."""
        self.logger = logging.getLogger(__name__)

        # Line detection parameters
        self.line_detection_params = {
            'rho': 1,                    # Distance resolution in pixels
            'theta': np.pi / 180,        # Angular resolution in radians
            'threshold': 50,             # Minimum votes for line detection
            'min_line_length': 50,       # Minimum line length
            'max_line_gap': 10           # Maximum gap between line segments
        }

        # Edge detection parameters
        self.edge_params = {
            'low_threshold': 50,
            'high_threshold': 150,
            'kernel_size': 3
        }

        # Junction classification thresholds
        self.classification_thresholds = {
            'angle_tolerance': 15,       # Degrees tolerance for perpendicular lines
            'min_road_width': 30,        # Minimum road width in pixels
            'junction_center_radius': 100 # Radius around center for junction analysis
        }

    def detect_junction_type(self, video_path, sample_frames=5):
        """
        Detect junction type from video by analyzing multiple frames.

        Args:
            video_path: Path to input video
            sample_frames: Number of frames to sample for analysis

        Returns:
            Junction type: '2-way', '3-way', '4-way', or 'Unknown'
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return 'Unknown'

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample frames evenly throughout the video
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

            junction_votes = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Analyze frame for junction type
                junction_type = self._analyze_frame_for_junction(frame)
                if junction_type != 'Unknown':
                    junction_votes.append(junction_type)

            cap.release()

            # Determine most common junction type
            if junction_votes:
                junction_counter = Counter(junction_votes)
                most_common = junction_counter.most_common(1)[0][0]
                return most_common
            else:
                return 'Unknown'

        except Exception as e:
            self.logger.error(f"Error detecting junction type: {str(e)}")
            return 'Unknown'

    def _analyze_frame_for_junction(self, frame):
        """
        Analyze a single frame to determine junction type.

        Args:
            frame: Input frame

        Returns:
            Junction type based on road line analysis
        """
        # Preprocess frame
        processed_frame = self._preprocess_frame(frame)

        # Detect edges
        edges = self._detect_edges(processed_frame)

        # Detect lines
        lines = self._detect_lines(edges)

        if lines is None or len(lines) < 2:
            return 'Unknown'

        # Analyze lines to determine junction type
        junction_type = self._classify_junction_from_lines(lines, frame.shape)

        return junction_type

    def _preprocess_frame(self, frame):
        """Preprocess frame for better line detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Enhance contrast
        enhanced = cv2.equalizeHist(blurred)

        return enhanced

    def _detect_edges(self, frame):
        """Detect edges using Canny edge detection."""
        edges = cv2.Canny(
            frame,
            self.edge_params['low_threshold'],
            self.edge_params['high_threshold'],
            apertureSize=self.edge_params['kernel_size']
        )

        return edges

    def _detect_lines(self, edges):
        """Detect lines using Hough Line Transform."""
        # Use Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            self.line_detection_params['rho'],
            self.line_detection_params['theta'],
            self.line_detection_params['threshold'],
            minLineLength=self.line_detection_params['min_line_length'],
            maxLineGap=self.line_detection_params['max_line_gap']
        )

        return lines

    def _classify_junction_from_lines(self, lines, frame_shape):
        """
        Classify junction type based on detected lines.

        Args:
            lines: Detected lines from Hough transform
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Junction type classification
        """
        height, width = frame_shape[:2]
        center_x, center_y = width // 2, height // 2

        # Convert lines to angle and filter by relevance
        road_lines = self._filter_road_lines(lines, center_x, center_y)

        if len(road_lines) < 2:
            return 'Unknown'

        # Group lines by angle
        angle_groups = self._group_lines_by_angle(road_lines)

        # Classify based on number of distinct angle groups
        num_directions = len(angle_groups)

        if num_directions >= 4:
            return '4-way'
        elif num_directions == 3:
            return '3-way'
        elif num_directions == 2:
            # Check if it's a proper intersection or just parallel roads
            if self._is_proper_intersection(angle_groups):
                return '2-way'
            else:
                return 'Unknown'
        else:
            return 'Unknown'

    def _filter_road_lines(self, lines, center_x, center_y):
        """Filter lines that are likely to represent roads."""
        road_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Distance from line to frame center
            line_center_x = (x1 + x2) / 2
            line_center_y = (y1 + y2) / 2
            distance_to_center = np.sqrt(
                (line_center_x - center_x)**2 + (line_center_y - center_y)**2
            )

            # Filter criteria for road lines
            if (length > self.line_detection_params['min_line_length'] and
                distance_to_center < self.classification_thresholds['junction_center_radius']):

                road_lines.append({
                    'line': line[0],
                    'length': length,
                    'angle': angle,
                    'distance_to_center': distance_to_center
                })

        return road_lines

    def _group_lines_by_angle(self, road_lines, angle_tolerance=None):
        """Group lines by similar angles."""
        if angle_tolerance is None:
            angle_tolerance = self.classification_thresholds['angle_tolerance']

        angle_groups = []

        for road_line in road_lines:
            angle = road_line['angle']

            # Normalize angle to 0-180 range (treat opposite directions as same road)
            normalized_angle = abs(angle) if abs(angle) <= 90 else 180 - abs(angle)

            # Find existing group or create new one
            group_found = False
            for group in angle_groups:
                group_angle = group['angle']
                if abs(normalized_angle - group_angle) <= angle_tolerance:
                    group['lines'].append(road_line)
                    # Update group angle to average
                    all_angles = [line['angle'] for line in group['lines']]
                    group['angle'] = np.mean([abs(a) if abs(a) <= 90 else 180 - abs(a) 
                                            for a in all_angles])
                    group_found = True
                    break

            if not group_found:
                angle_groups.append({
                    'angle': normalized_angle,
                    'lines': [road_line]
                })

        # Filter out groups with insufficient lines
        significant_groups = [group for group in angle_groups if len(group['lines']) >= 1]

        return significant_groups

    def _is_proper_intersection(self, angle_groups):
        """
        Check if two angle groups represent a proper intersection.

        Args:
            angle_groups: List of angle groups

        Returns:
            True if it's a proper intersection, False otherwise
        """
        if len(angle_groups) != 2:
            return False

        angle1 = angle_groups[0]['angle']
        angle2 = angle_groups[1]['angle']

        # Check if angles are roughly perpendicular
        angle_diff = abs(angle1 - angle2)
        perpendicular_threshold = self.classification_thresholds['angle_tolerance']

        return (abs(angle_diff - 90) <= perpendicular_threshold or 
                abs(angle_diff - 0) <= perpendicular_threshold)

    def visualize_junction_analysis(self, frame, lines=None, junction_type=None):
        """
        Create visualization of junction analysis.

        Args:
            frame: Input frame
            lines: Detected lines (optional)
            junction_type: Detected junction type (optional)

        Returns:
            Frame with junction analysis visualization
        """
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]

        # Draw detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center point
        center_x, center_y = width // 2, height // 2
        cv2.circle(vis_frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # Draw analysis radius
        radius = self.classification_thresholds['junction_center_radius']
        cv2.circle(vis_frame, (center_x, center_y), radius, (255, 0, 0), 1)

        # Add junction type label
        if junction_type:
            label = f"Junction: {junction_type}"
            cv2.putText(vis_frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return vis_frame

    def detect_road_directions(self, frame):
        """
        Detect the main road directions in the frame.

        Args:
            frame: Input frame

        Returns:
            List of road direction angles
        """
        # Preprocess and detect lines
        processed = self._preprocess_frame(frame)
        edges = self._detect_edges(processed)
        lines = self._detect_lines(edges)

        if lines is None:
            return []

        # Filter and group lines
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        road_lines = self._filter_road_lines(lines, center_x, center_y)
        angle_groups = self._group_lines_by_angle(road_lines)

        # Extract primary road directions
        directions = []
        for group in angle_groups:
            directions.append(group['angle'])

        return sorted(directions)

    def analyze_junction_complexity(self, frame):
        """
        Analyze the complexity of a junction.

        Args:
            frame: Input frame

        Returns:
            Dictionary with complexity metrics
        """
        # Detect lines and analyze
        processed = self._preprocess_frame(frame)
        edges = self._detect_edges(processed)
        lines = self._detect_lines(edges)

        if lines is None:
            return {'complexity': 'low', 'metrics': {}}

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        road_lines = self._filter_road_lines(lines, center_x, center_y)
        angle_groups = self._group_lines_by_angle(road_lines)

        # Calculate complexity metrics
        metrics = {
            'total_lines': len(lines),
            'road_lines': len(road_lines),
            'angle_groups': len(angle_groups),
            'average_line_length': np.mean([line['length'] for line in road_lines]) if road_lines else 0,
            'angle_variance': np.var([group['angle'] for group in angle_groups]) if angle_groups else 0
        }

        # Determine complexity level
        if len(angle_groups) >= 4:
            complexity = 'high'
        elif len(angle_groups) == 3:
            complexity = 'medium'
        elif len(angle_groups) == 2:
            complexity = 'low'
        else:
            complexity = 'minimal'

        return {
            'complexity': complexity,
            'metrics': metrics,
            'junction_type': self._classify_junction_from_lines(lines, frame.shape)
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the junction detector
    detector = JunctionDetector()

    # Test with a sample frame (would need actual video/image)
    # junction_type = detector.detect_junction_type("sample_video.mp4")
    # print(f"Detected junction type: {junction_type}")

    print("Junction Detector module loaded successfully!")
