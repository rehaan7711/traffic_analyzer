import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import cv2

class CentroidTracker:
    """
    Centroid tracking algorithm for tracking multiple objects across video frames.
    Based on computing centroids of bounding boxes and associating them between frames.
    """

    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize the centroid tracker.

        Args:
            max_disappeared: Maximum number of frames object can disappear before deregistering
            max_distance: Maximum distance for associating detections with existing objects
        """
        # Initialize counters and dictionaries
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # Configuration parameters
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Additional tracking information
        self.object_history = OrderedDict()  # Store trajectory history
        self.object_speeds = OrderedDict()   # Store speed information
        self.frame_count = 0

    def register(self, centroid, bbox=None):
        """
        Register a new object with the next available object ID.

        Args:
            centroid: (x, y) coordinates of object center
            bbox: Optional bounding box [x1, y1, x2, y2]

        Returns:
            object_id: ID of the registered object
        """
        # Register object with next available ID
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0

        # Initialize tracking history
        self.object_history[self.next_object_id] = [centroid]
        self.object_speeds[self.next_object_id] = 0.0

        # Increment object ID for next object
        object_id = self.next_object_id
        self.next_object_id += 1

        return object_id

    def deregister(self, object_id):
        """
        Deregister an object by removing it from all tracking dictionaries.

        Args:
            object_id: ID of object to deregister
        """
        # Remove object from all tracking dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_history[object_id]
        del self.object_speeds[object_id]

    def update(self, detections):
        """
        Update the tracker with new detections.

        Args:
            detections: List of detection dictionaries with 'bbox' key

        Returns:
            Dictionary of {object_id: centroid} for current frame
        """
        self.frame_count += 1

        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Deregister objects that have disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # Initialize array of input centroids for current frame
        input_centroids = np.zeros((len(detections), 2), dtype="int")

        # Extract centroids from detections
        for (i, detection) in enumerate(detections):
            bbox = detection['bbox']
            cx = int((bbox[0] + bbox[2]) / 2.0)
            cy = int((bbox[1] + bbox[3]) / 2.0)
            input_centroids[i] = (cx, cy)

        # If no existing objects, register all detections as new objects
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])

        # Otherwise, try to match existing objects with new detections
        else:
            # Grab set of object centroids and convert to array
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Compute distance matrix between existing objects and new detections
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Keep track of used row and column indices
            used_rows = set()
            used_cols = set()

            # Loop over the combination of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # Ignore if already examined
                if row in used_rows or col in used_cols:
                    continue

                # Check if distance is acceptable
                if D[row, col] > self.max_distance:
                    continue

                # Update object position
                object_id = object_ids[row]
                old_centroid = self.objects[object_id]
                new_centroid = input_centroids[col]

                self.objects[object_id] = new_centroid
                self.disappeared[object_id] = 0

                # Update tracking history
                self._update_history(object_id, old_centroid, new_centroid)

                # Mark this row and column as used
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # If more objects than detections, mark objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # Deregister if disappeared too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # If more detections than objects, register new objects
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

    def _update_history(self, object_id, old_centroid, new_centroid):
        """
        Update tracking history and calculate speed for an object.

        Args:
            object_id: Object ID
            old_centroid: Previous centroid position
            new_centroid: New centroid position
        """
        # Add new centroid to history
        self.object_history[object_id].append(new_centroid)

        # Limit history length to prevent memory growth
        max_history_length = 30
        if len(self.object_history[object_id]) > max_history_length:
            self.object_history[object_id] = self.object_history[object_id][-max_history_length:]

        # Calculate speed (pixels per frame)
        distance = np.sqrt((new_centroid[0] - old_centroid[0])**2 + 
                          (new_centroid[1] - old_centroid[1])**2)
        self.object_speeds[object_id] = distance

    def get_object_trajectory(self, object_id, max_points=10):
        """
        Get trajectory points for an object.

        Args:
            object_id: Object ID
            max_points: Maximum number of trajectory points to return

        Returns:
            List of trajectory points
        """
        if object_id not in self.object_history:
            return []

        history = self.object_history[object_id]
        return history[-max_points:] if len(history) > max_points else history

    def get_object_speed(self, object_id):
        """
        Get current speed for an object.

        Args:
            object_id: Object ID

        Returns:
            Speed in pixels per frame
        """
        return self.object_speeds.get(object_id, 0.0)

    def get_object_direction(self, object_id, window_size=5):
        """
        Calculate movement direction for an object.

        Args:
            object_id: Object ID
            window_size: Number of recent points to use for direction calculation

        Returns:
            Direction angle in degrees (0-360), or None if insufficient data
        """
        if object_id not in self.object_history:
            return None

        history = self.object_history[object_id]

        if len(history) < 2:
            return None

        # Use recent points for direction calculation
        recent_points = history[-window_size:] if len(history) >= window_size else history

        if len(recent_points) < 2:
            return None

        # Calculate direction from first to last point in window
        start_point = recent_points[0]
        end_point = recent_points[-1]

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Calculate angle in degrees
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # Convert to 0-360 range
        if angle < 0:
            angle += 360

        return angle

    def draw_trajectories(self, frame, trajectory_length=10, thickness=2):
        """
        Draw object trajectories on frame.

        Args:
            frame: Input frame
            trajectory_length: Number of trajectory points to draw
            thickness: Line thickness

        Returns:
            Frame with trajectories drawn
        """
        output_frame = frame.copy()

        # Draw trajectory for each tracked object
        for object_id in self.objects.keys():
            trajectory = self.get_object_trajectory(object_id, trajectory_length)

            if len(trajectory) < 2:
                continue

            # Choose color based on object ID
            color = self._get_trajectory_color(object_id)

            # Draw trajectory lines
            for i in range(len(trajectory) - 1):
                pt1 = tuple(map(int, trajectory[i]))
                pt2 = tuple(map(int, trajectory[i + 1]))
                cv2.line(output_frame, pt1, pt2, color, thickness)

            # Draw direction arrow at current position
            if len(trajectory) >= 2:
                current_pos = trajectory[-1]
                prev_pos = trajectory[-2]

                # Calculate direction vector
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]

                # Normalize and scale for arrow
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx = dx / length * 20
                    dy = dy / length * 20

                    # Draw arrow
                    arrow_tip = (int(current_pos[0] + dx), int(current_pos[1] + dy))
                    cv2.arrowedLine(output_frame, 
                                   tuple(map(int, current_pos)), 
                                   arrow_tip, 
                                   color, thickness)

        return output_frame

    def _get_trajectory_color(self, object_id):
        """Get color for trajectory based on object ID."""
        # Generate consistent color for each object ID
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

        return colors[object_id % len(colors)]

    def get_statistics(self):
        """
        Get tracking statistics.

        Returns:
            Dictionary with tracking statistics
        """
        stats = {
            'active_objects': len(self.objects),
            'total_objects_seen': self.next_object_id,
            'frame_count': self.frame_count,
            'average_speed': 0.0,
            'objects_by_speed': {
                'stationary': 0,  # speed < 1
                'slow': 0,        # 1 <= speed < 5
                'medium': 0,      # 5 <= speed < 15
                'fast': 0         # speed >= 15
            }
        }

        if len(self.object_speeds) > 0:
            speeds = list(self.object_speeds.values())
            stats['average_speed'] = np.mean(speeds)

            # Categorize by speed
            for speed in speeds:
                if speed < 1:
                    stats['objects_by_speed']['stationary'] += 1
                elif speed < 5:
                    stats['objects_by_speed']['slow'] += 1
                elif speed < 15:
                    stats['objects_by_speed']['medium'] += 1
                else:
                    stats['objects_by_speed']['fast'] += 1

        return stats

    def reset(self):
        """Reset the tracker to initial state."""
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.object_history = OrderedDict()
        self.object_speeds = OrderedDict()
        self.frame_count = 0

    def export_trajectories(self, output_path):
        """
        Export tracking trajectories to JSON file.

        Args:
            output_path: Output file path
        """
        import json
        from datetime import datetime

        # Prepare trajectory data
        trajectory_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'trajectories': {}
        }

        # Export each object's trajectory
        for object_id, history in self.object_history.items():
            trajectory_data['trajectories'][str(object_id)] = {
                'points': [list(point) for point in history],
                'current_speed': self.object_speeds.get(object_id, 0.0),
                'direction': self.get_object_direction(object_id),
                'active': object_id in self.objects
            }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

# Example usage and testing
if __name__ == "__main__":
    # Test the centroid tracker
    tracker = CentroidTracker()

    # Simulate some detections
    test_detections = [
        {'bbox': [100, 100, 150, 150]},
        {'bbox': [200, 200, 250, 250]},
    ]

    # Update tracker
    objects = tracker.update(test_detections)
    print(f"Tracked objects: {objects}")

    # Get statistics
    stats = tracker.get_statistics()
    print(f"Tracking statistics: {stats}")

    print("Centroid Tracker module loaded successfully!")
